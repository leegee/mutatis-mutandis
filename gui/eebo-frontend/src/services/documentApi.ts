export const showDocument =
    (corpus: string, docId: string, _tokenId?: number) =>
        window.open(`/api/doc/${ corpus }/${ docId }`, "_blank", "noopener,noreferrer");
