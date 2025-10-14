import type React from "react"
import type { Metadata } from "next"
import { GeistSans } from "geist/font/sans"
import { GeistMono } from "geist/font/mono"
import { Analytics } from "@vercel/analytics/next"
import { Suspense } from "react"
import "./globals.css"
import { LayoutWrapper } from "@/components/layout-wrapper"
import { AuthProvider } from "@/contexts/auth-context"
import { ModalProvider } from "@/contexts/modal-context"

export const metadata: Metadata = {
  title: "GeoVision - See the Earth. Understand the depth.",
  description: "Professional geological analysis platform for creating enhanced geological sections",
  generator: "v0.app",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className={`font-sans ${GeistSans.variable} ${GeistMono.variable}`}>
        <AuthProvider>
          <ModalProvider>
            <Suspense fallback={<div>Loading...</div>}>
              <LayoutWrapper>{children}</LayoutWrapper>
              <Analytics />
            </Suspense>
          </ModalProvider>
        </AuthProvider>
      </body>
    </html>
  )
}
