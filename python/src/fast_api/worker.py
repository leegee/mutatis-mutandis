import asyncio
from fast_api.jobs_dao import claim_next_job
from fast_api.state import init_state, STATE
from fast_api.runner import run_job

from lib.macberth import get_macberth_embedder
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from lib.eebo_logging import logger
import nltk

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
                await asyncio.to_thread(
                    run_topic_analysis_job,
                    job_id=job["job_id"],
                    payload=job.get("payload", {})
                )
            else:
                await asyncio.to_thread(
                    lambda: run_job(
                        job_id=job["job_id"],
                        concept=job["concept"],
                        index=STATE.index,
                        lookup=STATE.get_tier1_zarr_lookup(),
                    )
                )
        except Exception as e:
            logger.error(f"Worker failed on job {job.get('job_id')}: {e}")
            # TODO ? mark job as error in DAO


def run_topic_analysis_job(job_id: str, payload: dict):
    """Synchronous wrapper for BERTopic + MacBERTh job."""
    try:
        logger.info(f"Starting topic analysis for job {job_id}")

        documents = payload.get("documents", [])
        concept = payload.get("concept", "unknown")
        min_topic_size = payload.get("min_topic_size", 5)
        use_sentence_chunking = payload.get("use_sentence_chunking", False)

        if not documents:
            raise ValueError("No documents provided")

        if use_sentence_chunking:
            nltk.download('punkt', quiet=True)
            from nltk.tokenize import sent_tokenize

            documents = [
                sent.strip()
                for doc in documents
                for sent in sent_tokenize(doc)
                if len(sent.strip()) >= 30
            ]
            logger.info(f"Chunked into {len(documents)} sentences")

        # Load MacBERTh embedder (reuses your lib)
        embedder = get_macberth_embedder(pooling="mean")

        # BERTopic setup
        representation_model = KeyBERTInspired(top_n_words=12)

        topic_model = BERTopic(
            embedding_model=embedder,
            representation_model=representation_model,
            min_topic_size=min_topic_size,
            n_gram_range=(1, 2),
            calculate_probabilities=True,
            verbose=False
        )

        # Run the model
        topics, probs = topic_model.fit_transform(documents)

        result = {
            "status": "done",
            "concept": concept,
            "document_count": len(documents),
            "topics": topics.tolist(),
            "probabilities": probs.tolist() if probs is not None else None,
            "topic_info": topic_model.get_topic_info().to_dict(orient="records"),
            "representative_docs": topic_model.get_representative_docs(),
            # Optional: 2D embeddings for semantic map
            # "embeddings_2d":  _model.umap_model.transform(...).tolist()
        }

        # TODO: Save result via DAO / file / DB
        # save_topic_result(job_id, result)

        logger.info(f"Topic analysis completed for job {job_id} — {len(result['topic_info'])} topics found")
        return result

    except Exception as e:
        logger.error(f"Topic analysis failed for job {job_id}: {e}")
        # Mark job as error in DAO
        raise
