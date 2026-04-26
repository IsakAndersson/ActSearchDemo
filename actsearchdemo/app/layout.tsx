import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ActSearch Demo",
  description: "Password-gated Next.js UI for Docplus search",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" data-theme="corporate">
      <body className="antialiased">{children}</body>
    </html>
  );
}
