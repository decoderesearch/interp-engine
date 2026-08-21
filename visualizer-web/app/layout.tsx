import type { Metadata, Viewport } from "next";
import { Inter, Geist_Mono } from "next/font/google";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

// `adjustFontFallback` is off and the fallbacks are named by hand because the face
// Next generates otherwise is `src: local(Arial)` with Geist Mono's metrics laid
// over it — proportional, and on a machine whose fontconfig aliases Arial to a
// serif substitute it renders as a serif. It also sits *ahead* of any stack named
// in CSS, so no `--font-mono` tail can outvote it. Nearly every string in this app
// is an identifier read column by column, so a matched line height is worth less
// here than staying monospace while the webfont loads or when it fails to.
const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  adjustFontFallback: false,
  fallback: [
    "ui-monospace",
    "SFMono-Regular",
    "Menlo",
    "Consolas",
    "monospace",
  ],
});

const title = "Interp Engine";
const description =
  "interp-engine is a fast, standardized, and easy-to-use interpretability engine.";
const ogImage = {
  url: "https://neuronpedia.s3.amazonaws.com/site-assets/interp-engine-meta.png",
  width: 1730,
  height: 882,
  alt: title,
};

export const metadata: Metadata = {
  title,
  description,
  openGraph: {
    title,
    description,
    type: "website",
    siteName: title,
    images: [ogImage],
  },
  twitter: {
    card: "summary_large_image",
    title,
    description,
    images: [ogImage],
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="flex min-h-full flex-col overscroll-none">
        <TooltipProvider delayDuration={120}>{children}</TooltipProvider>
      </body>
    </html>
  );
}
