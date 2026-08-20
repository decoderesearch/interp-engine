import { useThemeConfig } from "@docusaurus/theme-common";
import useBaseUrl from "@docusaurus/useBaseUrl";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import ThemedImage from "@theme/ThemedImage";
import type { ReactNode } from "react";

/**
 * The navbar brand, pointing at `/` — the visualizer this site is mounted
 * inside — rather than at `/docs`, which the sidebar's first entry already
 * reaches.
 *
 * Swizzled rather than configured, because `navbar.logo.href` cannot say it.
 * `@theme/Logo` passes that href through `useBaseUrl`, which prepends
 * `baseUrl` to anything root-relative, so `/` arrives as `/docs/` — a link
 * from this site back to itself. `pathname://` does not help either:
 * `@docusaurus/Link` strips the prefix *before* adding the base url, then
 * treats what is left as external and opens it in a new tab.
 *
 * A plain `<a>` for the same reason, and it is the load-bearing part. `/` is
 * not a route in this SPA, so `@docusaurus/Link` would client-side navigate
 * to it and land on this site's own 404 page. Leaving the SPA needs a real
 * page load.
 */
export default function NavbarLogo(): ReactNode {
  const { siteConfig } = useDocusaurusContext();
  const {
    navbar: { title, logo },
  } = useThemeConfig();

  // Unconditional because hooks are, and free when there is no logo:
  // `useBaseUrl("")` returns "".
  const light = useBaseUrl(logo?.src ?? "");
  const dark = useBaseUrl(logo?.srcDark ?? logo?.src ?? "");

  return (
    <a className="navbar__brand" href="/">
      {logo && (
        <div className="navbar__logo">
          <ThemedImage
            className={logo.className}
            sources={{ light, dark }}
            height={logo.height}
            width={logo.width}
            // Empty marks the image decorative, which it is whenever the
            // title beside it already names the link.
            alt={logo.alt ?? (title ? "" : siteConfig.title)}
            style={logo.style}
          />
        </div>
      )}
      {title != null && <b className="navbar__title text--truncate">{title}</b>}
    </a>
  );
}
