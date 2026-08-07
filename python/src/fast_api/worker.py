import asyncio
import json
import nltk
from bertopic import BERTopic
from umap import UMAP

from lib.corpus_logging import logger
from lib.json import sanitize

from fast_api.models import embedder, representation_model, make_umap
from fast_api.jobs_dao import claim_next_job
from fast_api.state import init_state, STATE
from fast_api.runner import run_job

async def worker_loop():
    # TODO temp disabled in dev
    # init_state()

    while True:
        job = claim_next_job()

        if job is None:
            await asyncio.sleep(1)
            continue

        job_type = job.get("job_type") or "default"

        logger.info(f"[db.worker] Routing job type raw={job.get('job_type')} resolved={job_type}")

        try:
            if job_type == "topic_analysis":
                payload = job.get("payload", {})
                if isinstance(payload, str):
                    payload = json.loads(payload)
                await asyncio.to_thread(
                    run_topic_analysis_job,
                    job_id    = job["job_id"],
                    payload   = payload
                )
            else:
                await asyncio.to_thread(
                    lambda: run_job(
                        job_id  = job["job_id"],
                        concept = job["concept"],
                        index   = STATE.index,
                        lookup  = STATE.get_tier1_zarr_lookup(),
                    )
                )
        except Exception as e:
            logger.exception(f"Worker failed on job {job.get('job_id')}")
            # TODO ? mark job as error in DAO
            raise


def run_topic_analysis_job(job_id: str, payload: dict):
    """Synchronous wrapper for BERTopic + MacBERTh job."""
    try:
        logger.info(f"Starting topic analysis for job {job_id}")

        documents = payload.get("documents", [])
        concept = payload.get("concept", "unknown")
        min_topic_size = payload.get("min_topic_size", 2)
        use_sentence_chunking = payload.get("use_sentence_chunking", False)

        if not documents:
            raise ValueError("No documents provided")

        if use_sentence_chunking:
            nltk.download('punkt', quiet=True)
            nltk.download("punkt_tab", quiet=True)
            from nltk.tokenize import sent_tokenize

            documents = [
                sent.strip()
                for doc in documents
                for sent in sent_tokenize(doc)
                if len(sent.strip()) >= 30
            ]
            logger.info(f"Chunked into {len(documents)} sentences")

        umap_model = make_umap(len(documents))

        topic_model = BERTopic(
            embedding_model         = embedder,
            representation_model    = representation_model,
            umap_model              = umap_model,
            min_topic_size          = min_topic_size,
            calculate_probabilities = True,
            verbose=False
        )

        # Run the model
        topics, probs = topic_model.fit_transform(documents)

        topic_info = topic_model.get_topic_info()
        rep_docs = topic_model.get_representative_docs()

        topics_structured = [
            {
                "topic_id": row["Topic"],
                "size": row["Count"],
                "keywords": topic_model.get_topic(row["Topic"]),
                "representative_docs": rep_docs.get(row["Topic"], [])
            }
            for _, row in topic_info.iterrows()
            if row["Topic"] != -1
        ]

        result = {
            "status": "done",
            "concept": concept,
            "document_count": len(documents),
            "topics": topics_structured,
            "probabilities": probs.tolist() if probs is not None else None,
        }

        logger.info(
            f"Topic analysis completed for job {job_id} — {len(result['topics'])} topics found"
        )

        logger.info(
            "Topic analysis result:\n%s",
            json.dumps(sanitize(result), indent=2, ensure_ascii=False)
        )

        return sanitize(result)

    except Exception as e:
        logger.error(f"Topic analysis failed for job {job_id}: {e}")
        # Mark job as error in DAO
        raise
