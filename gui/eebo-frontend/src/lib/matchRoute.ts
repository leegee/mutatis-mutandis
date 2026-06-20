export function matchRoute(
  pattern: string, path: string
) {
  const clean = (p: string) => p.split("?")[0]; // ignore optional marker
  const pParts = clean(pattern).split("/").filter(Boolean);
  const pathParts = path.split("/").filter(Boolean);
  if (pParts.length !== pathParts.length) return false;

  return pParts.every((p, i) => {
    if (p.startsWith(":")) return true;
    return p === pathParts[i];
  });
};
