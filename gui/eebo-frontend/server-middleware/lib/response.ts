import type { ServerResponse } from "http";

export function json(
  res: ServerResponse,
  status: number,
  payload: unknown,
): void {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(payload, null, 2));
}

export function text(res: ServerResponse, status: number, body: string): void {
  res.statusCode = status;
  res.setHeader("Content-Type", "text/plain");
  res.end(body);
}

export function redirect(
  res: ServerResponse,
  location: string,
  status = 302,
): void {
  res.statusCode = status;
  res.setHeader("Location", location);
  res.end();
}

export function serverError(res: ServerResponse, error: unknown): void {
  console.error(error);

  json(res, 500, {
    error: error instanceof Error ? error.message : String(error),
  });
}
