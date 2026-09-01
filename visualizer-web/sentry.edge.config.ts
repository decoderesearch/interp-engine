import * as Sentry from "@sentry/nextjs";

// Nothing here runs on the edge today. The file exists because `instrumentation.ts` is asked
// for a runtime and answers for both, so the first edge route or proxy added later reports
// without a second setup. See `instrumentation-client.ts` for the DSN and PII notes.
Sentry.init({
  dsn: "https://ab60e8b83e7907f1cc7b0e3150245e45@o4508638349950976.ingest.us.sentry.io/4512013478002688",

  environment: process.env.VERCEL_ENV ?? process.env.NODE_ENV,

  tracesSampleRate: process.env.NODE_ENV === "development" ? 1.0 : 0.1,
});
