from lib.macberth_instance import get_shared_embedder
from lib.agent_tools.semantic_search import SemanticSearchTool

from lib.eebo_faiss import load_index
from lib.your_document_module import load_documents


def main():
    embedder = get_shared_embedder()
    index = load_index()
    documents = load_documents()

    tool = SemanticSearchTool( embedder, index, documents, )

    results = tool.search(
        "the divine right of kings",
        limit=5,
    )

    for r in results:
        print(
            r.score,
            r.title,
        )


if __name__ == "__main__":
    main()
