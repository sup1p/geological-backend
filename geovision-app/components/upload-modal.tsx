"use client"

import type React from "react"
import { useState, useRef, useCallback } from "react"
import { X, Upload, ImageIcon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { createGeologicalSection } from "@/lib/api"
import { useToast } from "@/hooks/use-toast"
import { ImageAreaSelector } from "./image-area-selector-improved"
import { MapPointSelector } from "./map-point-selector-fixed"
import { UnauthorizedError } from "./ui/unauthorized-error"
import { cropAllAreas, validateAreas } from "@/lib/image-cropping"

interface UploadModalProps {
  isOpen: boolean
  onClose: () => void
  onSuccess?: () => void
}

interface Point {
  x: number
  y: number
}

interface Area {
  x: number
  y: number
  width: number
  height: number
  label: string
}

export function UploadModal({ isOpen, onClose, onSuccess }: UploadModalProps) {
  const [sourceImage, setSourceImage] = useState<File | null>(null)
  const [areas, setAreas] = useState<{
    map: Area | null
    legend: Area | null
    column: Area | null
  }>({
    map: null,
    legend: null,
    column: null
  })
  const [startPoint, setStartPoint] = useState<Point | null>(null)
  const [endPoint, setEndPoint] = useState<Point | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [showUnauthorizedError, setShowUnauthorizedError] = useState(false)

  const sourceInputRef = useRef<HTMLInputElement>(null)

  const { toast } = useToast()

  // Stable callback for point changes
  const handlePointsChange = useCallback((start: Point | null, end: Point | null) => {
    console.log('[UploadModal] Points changed:', { start, end })
    setStartPoint(start)
    setEndPoint(end)
  }, [])

  // Stable callback for area changes
  const handleAreasChange = useCallback((selectedAreas: { map: Area | null; legend: Area | null; column: Area | null }) => {
    setAreas(selectedAreas)
  }, [])

  const resetForm = () => {
    setSourceImage(null)
    setAreas({
      map: null,
      legend: null,
      column: null
    })
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

    if (!sourceImage) {
      toast({
        title: "Missing image",
        description: "Please upload an image first",
        variant: "destructive",
      })
      return
    }

    // Validate that all areas are selected
    const validation = validateAreas(areas)
    if (!validation.isValid) {
      toast({
        title: "Missing selections",
        description: `Please select: ${validation.missingAreas.join(', ')}`,
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
      // Crop the areas from the source image
      console.log("[Upload] Cropping areas from source image...")
      const croppedImages = await cropAllAreas(sourceImage, areas)

      if (!croppedImages.mapImage || !croppedImages.legendImage || !croppedImages.columnImage) {
        throw new Error("Failed to crop one or more areas from the image")
      }

      // Points are already in the cropped map coordinate system
      // No need to add map area offset since we're sending the cropped map
      const mapStartX = Math.round(startPoint.x)
      const mapStartY = Math.round(startPoint.y)
      const mapEndX = Math.round(endPoint.x)
      const mapEndY = Math.round(endPoint.y)

      console.log("[Upload] Creating geological section with cropped images...")
      console.log("[Upload] Cross-section coordinates (relative to cropped map):", {
        start: { x: mapStartX, y: mapStartY },
        end: { x: mapEndX, y: mapEndY },
        mapArea: areas.map,
        croppedMapSize: `${areas.map.width}x${areas.map.height}`
      })
      
      const blob = await createGeologicalSection(
        croppedImages.mapImage,
        croppedImages.legendImage,
        croppedImages.columnImage,
        mapStartX,
        mapStartY,
        mapEndX,
        mapEndY,
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
      
      const errorMessage = error instanceof Error ? error.message : "Failed to create geological section"
      
      // Check if it's an unauthorized error (only after refresh attempts failed)
      if (errorMessage.includes("UNAUTHORIZED")) {
        console.log("[Upload] Token refresh failed, showing unauthorized error")
        setShowUnauthorizedError(true)
        return
      }
      
      // Log other errors for debugging
      console.log("[Upload] Non-auth error:", errorMessage)
      
      toast({
        title: "Upload failed",
        description: errorMessage,
        variant: "destructive",
      })
    } finally {
      setIsUploading(false)
    }
  }

  if (!isOpen) return null

  // Show unauthorized error overlay if needed
  if (showUnauthorizedError) {
    return (
      <UnauthorizedError 
        message="Your session has expired. Please log in again to create geological sections."
        onClose={() => {
          setShowUnauthorizedError(false)
          handleClose()
        }}
      />
    )
  }

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
          <div className="space-y-4">
            <div>
              <Label htmlFor="source">Upload Geological Data Image</Label>
              <p className="text-sm text-muted-foreground mb-3">
                Upload an image containing your geological map, legend, and stratigraphic column. You'll select each area next.
              </p>
              <div
                onClick={() => sourceInputRef.current?.click()}
                className="flex flex-col items-center justify-center h-32 border-2 border-dashed border-border rounded-lg cursor-pointer hover:border-primary transition-colors"
              >
                {sourceImage ? (
                  <div className="text-center">
                    <ImageIcon className="h-8 w-8 mx-auto text-primary mb-2" />
                    <p className="text-xs text-muted-foreground truncate px-2">{sourceImage.name}</p>
                    <p className="text-xs text-muted-foreground">Click to change</p>
                  </div>
                ) : (
                  <div className="text-center">
                    <Upload className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
                    <p className="text-xs text-muted-foreground">Click to upload image</p>
                  </div>
                )}
              </div>
              <input
                ref={sourceInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => setSourceImage(e.target.files?.[0] || null)}
              />
            </div>
          </div>

          <div className="space-y-4">
            <Label>Area Selection</Label>
            <p className="text-sm text-muted-foreground">
              Select the three required areas from your uploaded image by clicking the buttons below and drawing rectangles on the image.
            </p>
            
            {/* Area selection status */}
            <div className="grid grid-cols-3 gap-2 mb-4">
              {[
                { key: 'map', label: 'Map', color: '#3B82F6' },
                { key: 'legend', label: 'Legend', color: '#EF4444' },
                { key: 'column', label: 'Column', color: '#10B981' }
              ].map(({ key, label, color }) => (
                <div
                  key={key}
                  className={`p-3 rounded-lg border-2 transition-all ${
                    areas[key as keyof typeof areas]
                      ? 'border-green-500 bg-green-50'
                      : 'border-gray-200 bg-gray-50'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <div 
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: color }}
                    />
                    <span className="text-sm font-medium">{label}</span>
                    {areas[key as keyof typeof areas] && (
                      <div className="ml-auto">
                        <div className="w-5 h-5 bg-green-500 rounded-full flex items-center justify-center">
                          <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                        </div>
                      </div>
                    )}
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    {areas[key as keyof typeof areas] ? 'Selected' : 'Not selected'}
                  </div>
                </div>
              ))}
            </div>
            
            <ImageAreaSelector 
              imageFile={sourceImage}
              onAreasChange={handleAreasChange}
            />
          </div>

          <div className="space-y-4">
            <Label>Cross-Section Line Selection</Label>
            <p className="text-sm text-muted-foreground">
              Click on the map area to select the start and end points for your cross-section line.
            </p>
            
            {/* Point selection status */}
            <div className="grid grid-cols-2 gap-2 mb-4">
              <div className={`p-3 rounded-lg border-2 transition-all ${
                startPoint ? 'border-blue-500 bg-blue-50' : 'border-gray-200 bg-gray-50'
              }`}>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-blue-500" />
                  <span className="text-sm font-medium">Start Point</span>
                  {startPoint && (
                    <div className="ml-auto">
                      <div className="w-5 h-5 bg-green-500 rounded-full flex items-center justify-center">
                        <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      </div>
                    </div>
                  )}
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  {startPoint ? `(${Math.round(startPoint.x)}, ${Math.round(startPoint.y)})` : 'Not selected'}
                </div>
              </div>
              
              <div className={`p-3 rounded-lg border-2 transition-all ${
                endPoint ? 'border-red-500 bg-red-50' : 'border-gray-200 bg-gray-50'
              }`}>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-red-500" />
                  <span className="text-sm font-medium">End Point</span>
                  {endPoint && (
                    <div className="ml-auto">
                      <div className="w-5 h-5 bg-green-500 rounded-full flex items-center justify-center">
                        <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      </div>
                    </div>
                  )}
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  {endPoint ? `(${Math.round(endPoint.x)}, ${Math.round(endPoint.y)})` : 'Not selected'}
                </div>
              </div>
            </div>

            <MapPointSelector 
              sourceImage={sourceImage}
              mapArea={areas.map}
              onPointsChange={handlePointsChange}
            />
          </div>

          {/* Overall completion status */}
          <div className="p-4 rounded-lg bg-muted border">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="font-medium text-sm">Upload Progress</h4>
                <div className="flex items-center gap-4 mt-2">
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${sourceImage ? 'bg-green-500' : 'bg-gray-300'}`} />
                    <span className="text-xs">Image</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${
                      areas.map && areas.legend && areas.column ? 'bg-green-500' : 'bg-gray-300'
                    }`} />
                    <span className="text-xs">Areas ({Object.values(areas).filter(Boolean).length}/3)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${startPoint && endPoint ? 'bg-green-500' : 'bg-gray-300'}`} />
                    <span className="text-xs">Points</span>
                  </div>
                </div>
              </div>
              <div className="text-right">
                {sourceImage && areas.map && areas.legend && areas.column && startPoint && endPoint ? (
                  <div className="flex items-center gap-1 text-green-600">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    <span className="text-xs font-medium">Ready to Upload</span>
                  </div>
                ) : (
                  <span className="text-xs text-muted-foreground">Complete all steps</span>
                )}
              </div>
            </div>
          </div>

          <div className="flex justify-end gap-2">
            <Button type="button" variant="outline" onClick={handleClose}>
              Cancel
            </Button>
            <Button 
              type="submit" 
              disabled={isUploading || !sourceImage || !areas.map || !areas.legend || !areas.column || !startPoint || !endPoint}
              className={
                sourceImage && areas.map && areas.legend && areas.column && startPoint && endPoint 
                  ? 'bg-green-600 hover:bg-green-700' 
                  : ''
              }
            >
              {isUploading ? "Creating Section..." : "Create Section"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
}
