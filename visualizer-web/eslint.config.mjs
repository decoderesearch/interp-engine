import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // The samples site: its own build, its own toolchain. `public/docs` is the
    // minified bundle it emits, which this config would otherwise lint as if
    // somebody had written it.
    "public/docs/**",
    "docs-site/**",
  ]),
]);

export default eslintConfig;
