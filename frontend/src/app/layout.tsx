import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "EchoDJ — AI Radio Station",
  description: "An agentic, personalized AI radio station built on the Spotify ecosystem. EchoDJ selects songs based on your deep listening history and speaks between tracks like a real DJ.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="theme-color" content="#0a0a0f" />
      </head>
      <body>
        {children}
      </body>
    </html>
  );
}
