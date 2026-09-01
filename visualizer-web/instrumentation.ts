import * as Sentry from "@sentry/nextjs";

export async function register() {
  if (process.env.NEXT_RUNTIME === "nodejs") {
    await import("./sentry.server.config");
  }

  if (process.env.NEXT_RUNTIME === "edge") {
    await import("./sentry.edge.config");
  }
}

// Server-side request errors — the API routes and any Server Component render — reach Sentry
// through this hook and no other. Without the export they are logged and dropped.
export const onRequestError = Sentry.captureRequestError;
