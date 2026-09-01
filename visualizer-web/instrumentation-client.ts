import * as Sentry from "@sentry/nextjs";

// The DSN is written out rather than read from an environment variable because it is public
// by design — it ships inside the client bundle either way, and it only grants the right to
// post events to this one project. An env var would buy nothing and add a way for the deploy
// to go quiet: unset in Vercel, `Sentry.init` becomes a no-op and nothing says so.
//
// Browser events go out through `/monitoring`, the tunnel `next.config.ts` asks
// `withSentryConfig` to mount. That is what keeps this working under the CSP there, whose
// `connect-src` names `'self'` and the Hub and nothing else — the ingest host is never
// contacted from the page. It also means an ad blocker's Sentry list cannot silence the
// reports, which is most of the point of collecting them from real readers.
Sentry.init({
  dsn: "https://ab60e8b83e7907f1cc7b0e3150245e45@o4508638349950976.ingest.us.sentry.io/4512013478002688",

  // Preview deploys and production share the project, so name the environment or the two
  // arrive indistinguishable. Vercel exposes this to the browser for Next projects; local
  // `next dev` falls through to "development".
  environment: process.env.NEXT_PUBLIC_VERCEL_ENV ?? process.env.NODE_ENV,

  // No `dataCollection` object at all, which is not the same as an empty one: passing `{}`
  // opts *in* to every category it does not mention. Left off, the SDK stays conservative —
  // no IP addresses, no request bodies. The questions readers type into Ask Riz are request
  // bodies.
  tracesSampleRate: process.env.NODE_ENV === "development" ? 1.0 : 0.1,
});

export const onRouterTransitionStart = Sentry.captureRouterTransitionStart;
