import type * as Preset from "@docusaurus/preset-classic";
import type { Config } from "@docusaurus/types";
import { themes } from "prism-react-renderer";

/**
 * The samples site, mounted at `/docs` inside the visualizer next door.
 *
 * `npm run build` writes to `../public/docs`, which the Next app serves as static
 * files — so `baseUrl` has to be `/docs/` and every asset URL is built from it.
 * `trailingSlash: false` is the part that makes the mount work: it emits
 * `intro.html` beside `intro/index.html`'s alternative, and a page-per-file is
 * what `next.config.ts`'s one rewrite can resolve.
 */
const config: Config = {
  title: "interp-engine",
  tagline: "Code samples",
  favicon: "ielogo.png",

  url: "https://interp-engine.org",
  baseUrl: "/docs/",
  trailingSlash: false,

  // Nothing here is generated from a source file, so a stale link is a typo and
  // should stop the build rather than ship.
  onBrokenLinks: "throw",
  markdown: { hooks: { onBrokenMarkdownLinks: "throw" } },

  presets: [
    [
      "classic",
      {
        docs: {
          // Docs-only: `/docs` is the samples index, with no landing page in front
          // of it. There is a landing page already, and it is the diagram.
          routeBasePath: "/",
          sidebarPath: "./sidebars.ts",
        },
        blog: false,
        pages: false,
        theme: { customCss: "./src/css/custom.css" },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    navbar: {
      // Both of these render inside `src/theme/Navbar/Logo`, which is swizzled
      // so the brand links to `/` rather than here. An `href` added below
      // would be read by the stock component and ignored by that one.
      title: "interp-engine",
      logo: { alt: "", src: "ielogo.png" },
      items: [
        // Raw HTML, because `/` cannot be expressed as a navbar `href`. Every
        // form of it — including the `pathname://` escape hatch — is resolved
        // against `baseUrl` on the way out and arrives as `/docs/`, a link from
        // this site back to itself. `theme/NavbarItem` appends the base url
        // *after* stripping that prefix, deliberately, so there is no href that
        // leaves this directory. This element is emitted verbatim.
        {
          type: "html",
          position: "right",
          value:
            '<a class="navbar__item navbar__link" href="/">Visualizer</a>',
        },
        {
          href: "https://github.com/decoderesearch/interp-engine",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "light",
      copyright:
        'Apache-2.0 · <a href="/">visualizer</a> · <a href="https://github.com/decoderesearch/interp-engine">GitHub</a>',
    },
    prism: {
      theme: themes.github,
      darkTheme: themes.dracula,
      additionalLanguages: ["bash", "python"],
    },
    colorMode: { defaultMode: "light", respectPrefersColorScheme: true },
  } satisfies Preset.ThemeConfig,
};

export default config;
