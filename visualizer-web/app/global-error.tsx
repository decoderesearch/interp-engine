"use client";

import * as Sentry from "@sentry/nextjs";
import NextError from "next/error";
import { useEffect } from "react";

// The whole app is one client-rendered page, so a render error here takes the diagram with it
// and there is no smaller boundary to catch it. `global-error.tsx` replaces the root layout
// when that happens, which is why it renders its own `<html>`.
export default function GlobalError({
  error,
}: {
  error: Error & { digest?: string };
}) {
  useEffect(() => {
    Sentry.captureException(error);
  }, [error]);

  return (
    <html lang="en">
      <body>
        {/* Next's own error page. Its type wants a status code and the App Router has none
        to give for a render error, so 0 selects the generic message. */}
        <NextError statusCode={0} />
      </body>
    </html>
  );
}
