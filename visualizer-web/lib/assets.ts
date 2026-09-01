/**
 * Assets this app does not serve itself.
 *
 * The screen recording on the tour's last slide, and the benchmark clip on the
 * Fast claim, both live on the bucket the rest of Neuronpedia's site assets do.
 * They are large and animated, so they are fetched only by the slide or card
 * that shows them.
 *
 * `next/image` is not what draws the GIF: optimizing it flattens it to a still,
 * and `unoptimized` on a remote source still wants the host declared in
 * `images.remotePatterns`. A plain `img` needs neither, and this is the only
 * cross-origin image in the app.
 *
 * `next.config.ts` imports the origin from here for `img-src` and `media-src`,
 * because a CSP that does not name this host fails silently: the slide renders,
 * the file does not, and the only sign of it is a console line nobody is reading.
 */
export const DEMO_GIF =
  "https://neuronpedia.s3.amazonaws.com/site-assets/interp-engine-demo.gif";

/** Its intrinsic size, so the slide reserves the space before it arrives. */
export const DEMO_GIF_WIDTH = 1280;
export const DEMO_GIF_HEIGHT = 843;

export const BENCHMARK_VIDEO =
  "https://neuronpedia.s3.amazonaws.com/site-assets/ie-benchmark-hi.mp4";

export const SIZER_VIDEO =
  "https://neuronpedia.s3.amazonaws.com/site-assets/gpu-sizer-2.mp4";

export const DEMO_GIF_ORIGIN = new URL(DEMO_GIF).origin;
