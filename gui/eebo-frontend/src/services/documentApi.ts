export const showDocument = (docId: string, _tokenId?: number) => window.open(`/api/doc/${ docId }`, "_blank", "noopener,noreferrer");
