import * as Sentry from "@sentry/nextjs";

// Loaded by `instrumentation.ts` on the Node runtime — the API routes and the prerender.
// See `instrumentation-client.ts` for why the DSN is a literal and why `dataCollection`
// is absent rather than empty.
Sentry.init({
  dsn: "https://ab60e8b83e7907f1cc7b0e3150245e45@o4508638349950976.ingest.us.sentry.io/4512013478002688",

  environment: process.env.VERCEL_ENV ?? process.env.NODE_ENV,

  tracesSampleRate: process.env.NODE_ENV === "development" ? 1.0 : 0.1,
});
