"use client"

import type React from "react"
import { useState, useRef } from "react"
import { X, Upload, ImageIcon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { createGeologicalSection } from "@/lib/api"
import { useToast } from "@/hooks/use-toast"
import { ImagePointSelector } from "./image-point-selector"

interface UploadModalProps {
  isOpen: boolean
  onClose: () => void
  onSuccess?: () => void
}

interface Point {
  x: number
  y: number
}

export function UploadModal({ isOpen, onClose, onSuccess }: UploadModalProps) {
  const [mapImage, setMapImage] = useState<File | null>(null)
  const [legendImage, setLegendImage] = useState<File | null>(null)
  const [columnImage, setColumnImage] = useState<File | null>(null)
  const [startPoint, setStartPoint] = useState<Point | null>(null)
  const [endPoint, setEndPoint] = useState<Point | null>(null)
  const [isUploading, setIsUploading] = useState(false)

  const mapInputRef = useRef<HTMLInputElement>(null)
  const legendInputRef = useRef<HTMLInputElement>(null)
  const columnInputRef = useRef<HTMLInputElement>(null)

  const { toast } = useToast()

  const resetForm = () => {
    setMapImage(null)
    setLegendImage(null)
    setColumnImage(null)
    setStartPoint(null)
    setEndPoint(null)
    setIsUploading(false)
  }

  const handleClose = () => {
    resetForm()
    onClose()
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!mapImage || !legendImage || !columnImage) {
      toast({
        title: "Missing files",
        description: "Please upload all required images",
        variant: "destructive",
      })
      return
    }

    if (!startPoint || !endPoint) {
      toast({
        title: "Missing coordinates",
        description: "Please select both start and end points on the map",
        variant: "destructive",
      })
      return
    }

    setIsUploading(true)

    try {
      const blob = await createGeologicalSection(
        mapImage,
        legendImage,
        columnImage,
        startPoint.x,
        startPoint.y,
        endPoint.x,
        endPoint.y,
      )

      // Download the result
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `geological_section_${Date.now()}.png`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)

      toast({
        title: "Success",
        description: "Geological section created successfully",
      })

      onSuccess?.()
      resetForm()
      onClose()
    } catch (error) {
      console.error("[v0] Upload error:", error)
      toast({
        title: "Upload failed",
        description: error instanceof Error ? error.message : "Failed to create geological section",
        variant: "destructive",
      })
    } finally {
      setIsUploading(false)
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={handleClose} />

      <div className="relative z-10 w-full max-w-6xl rounded-lg border border-border bg-card shadow-lg max-h-[95vh] flex flex-col">
        <div className="flex items-center justify-between border-b border-border p-4">
          <h2 className="text-lg font-semibold">Upload Geological Data</h2>
          <Button variant="ghost" size="icon" onClick={handleClose}>
            <X className="h-5 w-5" />
            <span className="sr-only">Close</span>
          </Button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-6 overflow-y-auto flex-1">
          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <Label htmlFor="map">Geological Map</Label>
              <div
                onClick={() => mapInputRef.current?.click()}
                className="flex flex-col items-center justify-center h-32 border-2 border-dashed border-border rounded-lg cursor-pointer hover:border-primary transition-colors"
              >
                {mapImage ? (
                  <div className="text-center">
                    <ImageIcon className="h-8 w-8 mx-auto text-primary mb-2" />
                    <p className="text-xs text-muted-foreground truncate px-2">{mapImage.name}</p>
                  </div>
                ) : (
                  <div className="text-center">
                    <Upload className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
                    <p className="text-xs text-muted-foreground">Click to upload</p>
                  </div>
                )}
              </div>
              <input
                ref={mapInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => setMapImage(e.target.files?.[0] || null)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="legend">Legend</Label>
              <div
                onClick={() => legendInputRef.current?.click()}
                className="flex flex-col items-center justify-center h-32 border-2 border-dashed border-border rounded-lg cursor-pointer hover:border-primary transition-colors"
              >
                {legendImage ? (
                  <div className="text-center">
                    <ImageIcon className="h-8 w-8 mx-auto text-primary mb-2" />
                    <p className="text-xs text-muted-foreground truncate px-2">{legendImage.name}</p>
                  </div>
                ) : (
                  <div className="text-center">
                    <Upload className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
                    <p className="text-xs text-muted-foreground">Click to upload</p>
                  </div>
                )}
              </div>
              <input
                ref={legendInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => setLegendImage(e.target.files?.[0] || null)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="column">Stratigraphic Column</Label>
              <div
                onClick={() => columnInputRef.current?.click()}
                className="flex flex-col items-center justify-center h-32 border-2 border-dashed border-border rounded-lg cursor-pointer hover:border-primary transition-colors"
              >
                {columnImage ? (
                  <div className="text-center">
                    <ImageIcon className="h-8 w-8 mx-auto text-primary mb-2" />
                    <p className="text-xs text-muted-foreground truncate px-2">{columnImage.name}</p>
                  </div>
                ) : (
                  <div className="text-center">
                    <Upload className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
                    <p className="text-xs text-muted-foreground">Click to upload</p>
                  </div>
                )}
              </div>
              <input
                ref={columnInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => setColumnImage(e.target.files?.[0] || null)}
              />
            </div>
          </div>

          <div className="space-y-4">
            <Label>Cross-Section Line Selection</Label>
            <p className="text-sm text-muted-foreground">
              Click on the geological map to select the start and end points for your cross-section line.
            </p>
            <ImagePointSelector 
              imageFile={mapImage}
              onPointsChange={(start, end) => {
                setStartPoint(start)
                setEndPoint(end)
              }}
            />
          </div>

          <div className="flex justify-end gap-2">
            <Button type="button" variant="outline" onClick={handleClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={isUploading}>
              {isUploading ? "Creating Section..." : "Create Section"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
}
