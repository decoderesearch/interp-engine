import type { NextConfig } from "next";
import { withSentryConfig } from "@sentry/nextjs";

import { DEMO_GIF_ORIGIN } from "./lib/assets";
import { HUB_ORIGINS } from "./lib/hub";

// `script-src` and `style-src` carry `'unsafe-inline'` because every page here is prerendered at
// build time. The nonce alternative wants middleware to mint one per request, which makes each
// route dynamic — a server rendering a diagram whose inputs are all client state. So every other
// directive stays at `'self'` and inline injection is the only hole left open.
//
// `img-src` and `media-src` are the exceptions, and they name one host: the
// screen recording on the welcome tour's last slide and the benchmark clip on
// the Fast claim, both on Neuronpedia's asset bucket rather than in this
// repository. The origin is imported rather than typed out because a CSP that
// does not name it fails silently — see `lib/assets.ts`.
//
// `connect-src` is the third, and it names both halves of how the GPU finder reaches the Hub. Most
// lookups go through `/api/hub` on this project's token — that is the `'self'` — but a reader who
// supplies their own token is sent straight to huggingface.co instead, because `lib/hub.ts` promises
// that token is sent there and nowhere else and a proxy would retract it. The origins come from
// `lib/hub.ts`, which explains why the CDN wildcards are needed as well as the apex: a redirect
// chain is checked against the policy at every hop. Sentry rides on the same `'self'` and asks
// for nothing more, by way of the tunnel route at the foot of this file.
//
// `headers()` applies in development too, and React's dev build calls `eval` to rebuild callstacks
// across environments. Without the extra source below, `next dev` logs an eval refusal on every
// render and the RSC payload fetch fails, so hot reload falls back to a full browser navigation.
// React never calls `eval` in a production build, so this widens nothing about what ships.
const devOnly = process.env.NODE_ENV === "development" ? " 'unsafe-eval'" : "";

const csp = [
  "default-src 'self'",
  `script-src 'self' 'unsafe-inline'${devOnly}`,
  "style-src 'self' 'unsafe-inline'",
  `img-src 'self' data: ${DEMO_GIF_ORIGIN}`,
  `media-src 'self' ${DEMO_GIF_ORIGIN}`,
  "font-src 'self'",
  `connect-src 'self' ${HUB_ORIGINS.join(" ")}`,
  "frame-ancestors 'none'",
  "base-uri 'self'",
  "form-action 'self'",
  "object-src 'none'",
].join("; ");

const nextConfig: NextConfig = {
  poweredByHeader: false,
  // The samples site is a separate Docusaurus build (`docs-site/`) that writes into
  // `public/docs`, so `/docs/steering.html` is already served as a static file. These
  // two rules serve the extensionless URL its own navbar links to.
  //
  // `afterFiles` is the placement that makes this safe: it runs *after* the public
  // directory is checked, so every real file — `/docs/assets/js/main.js`,
  // `/docs/ielogo.png`, `/docs/sitemap.xml` — is served as itself and never rewritten.
  // Only a path with no file behind it reaches the rules below.
  //
  // One segment, because the built pages are flat: a nested doc id would emit
  // `a/b.html` and need a second rule. `sidebars.ts` says so where the ids are written.
  rewrites() {
    return {
      beforeFiles: [],
      afterFiles: [
        { source: "/docs", destination: "/docs/index.html" },
        { source: "/docs/:page", destination: "/docs/:page.html" },
        // The GPU sizer's model lives in the path so the URL *is* the repo id --
        // see `lib/sizer-link.ts`. A rewrite rather than a route: this app is one
        // prerendered page, and a dynamic segment carrying a string would make it
        // server-rendered to learn something the client reads off `location`
        // anyway. Every `/sizer/...` serves the same HTML `/` does.
        { source: "/sizer", destination: "/" },
        { source: "/sizer/:model*", destination: "/" },
      ],
      fallback: [],
    };
  },
  headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "Content-Security-Policy", value: csp },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          {
            key: "Permissions-Policy",
            value:
              "camera=(), microphone=(), geolocation=(), browsing-topics=()",
          },
          // Deliberately no `includeSubDomains` and no `preload`: this is one page on one
          // hostname, and both of those bind every sibling subdomain of the custom domain to
          // HTTPS for two years. Add them from the domain that owns the apex, not from here.
          { key: "Strict-Transport-Security", value: "max-age=63072000" },
        ],
      },
    ];
  },
};

// `tunnelRoute` is what lets Sentry report from the browser at all: the CSP above allows
// `connect-src 'self'`, and the plugin mounts `/monitoring` on this origin to forward events
// to the ingest host server-side. Naming the ingest host in the policy instead would work and
// then lose the reports of every reader running an ad blocker, since those lists carry it.
//
// `authToken` is the only secret here, it is only read at build time, and without it the
// build still succeeds — it just uploads no source maps, so production stack traces stay
// minified. See `.env.example`.
export default withSentryConfig(nextConfig, {
  org: "johnny-66",
  project: "interp-engine-visualizer",
  authToken: process.env.SENTRY_AUTH_TOKEN,
  widenClientFileUpload: true,
  tunnelRoute: "/monitoring",
  silent: !process.env.CI,
});
