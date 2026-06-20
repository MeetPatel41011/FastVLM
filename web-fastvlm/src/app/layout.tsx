import type { Metadata } from "next";
import ParticleBackground from "@/components/ParticleBackground";
import "./globals.css";

export const metadata: Metadata = {
  title: "FastVLM - Local AI Q&A",
  description: "Live camera input Q&A system running 100% locally.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Google+Sans+Flex:opsz,slnt,wdth,wght,ROND@8..144,-10..0,25..150,400..500,0..100&display=swap" />
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Google+Sans+Code:ital,wght@0,400;1,400&display=swap" />
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Google+Symbols:opsz,wght,FILL,GRAD,ROND@40..48,300,0..1,0,50&display=block" />
      </head>
      <body suppressHydrationWarning>
        <ParticleBackground />
        <div style={{ position: 'relative', zIndex: 1 }}>
          {children}
        </div>
      </body>
    </html>
  );
}
