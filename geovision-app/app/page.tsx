"use client"

import { Upload } from "lucide-react"
import { useRouter } from "next/navigation"
import { Card } from "@/components/ui/card"
import { useAuth } from "@/contexts/auth-context"
import { useModal } from "@/contexts/modal-context"

export default function HomePage() {
  const router = useRouter()
  const { isAuthenticated } = useAuth()
  const { openUploadModal } = useModal()

  const handleUploadClick = () => {
    if (!isAuthenticated) {
      router.push("/login")
    } else {
      openUploadModal()
    }
  }

  return (
    <div className="min-h-screen">

      <main className="container px-4 py-12">
        <div className="flex flex-col items-center space-y-8">
          {/* Main title */}
          <div className="text-center">
            <h1 className="text-4xl font-bold leading-tight text-balance">
              GeoVision: See the Earth. Understand the depth.
            </h1>
          </div>

          {/* Upload section */}
          <Card
            onClick={handleUploadClick}
            className="flex flex-col items-center justify-center p-6 bg-accent/20 border-2 border-accent hover:bg-accent/30 transition-colors cursor-pointer"
          >
            <Upload className="h-8 w-8 text-accent-foreground mb-2" />
            <h2 className="text-lg font-semibold text-accent-foreground">Upload Map</h2>
          </Card>

          {/* Results section */}
          <div className="text-center space-y-6">
            <h2 className="text-2xl font-semibold">And get results</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl">
              <img
                src="/example.jpg"
                alt="Geological cross-section example 1"
                className="w-full rounded-lg shadow-lg"
              />
              <img
                src="/example2.jpg"
                alt="Geological cross-section example 2"
                className="w-full rounded-lg shadow-lg"
              />
              <img
                src="/example3.jpg"
                alt="Geological cross-section example 3"
                className="w-full rounded-lg shadow-lg"
              />
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
