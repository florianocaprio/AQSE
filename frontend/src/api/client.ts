export class ApiError extends Error {
  readonly status: number;
  readonly details: unknown;

  constructor(message: string, status: number, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}
type RequestOptions = Omit<RequestInit, "body"> & {
  body?: unknown;
};

export async function requestJson<T>(
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const headers = new Headers(options.headers);
  let body: BodyInit | undefined;

  if (options.body !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(options.body);
  }

  const response = await fetch(path, { ...options, headers, body });
  if (!response.ok) {
    throw await apiErrorFromResponse(response);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

async function apiErrorFromResponse(response: Response): Promise<ApiError> {
  let details: unknown;
  try {
    details = await response.json();
  } catch {
    details = undefined;
  }

  const message = getErrorMessage(details) ?? `Request failed with HTTP ${response.status}.`;
  return new ApiError(message, response.status, details);
}

function getErrorMessage(payload: unknown): string | null {
  if (!payload || typeof payload !== "object" || !("detail" in payload)) {
    return null;
  }

  const detail = (payload as { detail: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (!Array.isArray(detail)) return null;

  const messages = detail.flatMap((item) => {
    if (!item || typeof item !== "object" || !("msg" in item)) return [];
    const message = (item as { msg: unknown }).msg;
    return typeof message === "string" ? [message] : [];
  });
  return messages.length > 0 ? messages.join(" ") : null;
}

export function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}
