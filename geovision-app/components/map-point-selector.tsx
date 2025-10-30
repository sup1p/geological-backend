"use client"

import React, { useState, useRef, useCallback, useEffect } from "react"
import { cn } from "@/lib/utils"

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

interface MapPointSelectorProps {
  sourceImage: File | null
  mapArea: Area | null
  onPointsChange: (startPoint: Point | null, endPoint: Point | null) => void
  className?: string
}

export function MapPointSelector({ sourceImage, mapArea, onPointsChange, className }: MapPointSelectorProps) {
  const [startPoint, setStartPoint] = useState<Point | null>(null)
  const [endPoint, setEndPoint] = useState<Point | null>(null)
  const [croppedMapUrl, setCroppedMapUrl] = useState<string | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)

  // Create cropped map image when source image or map area changes
  useEffect(() => {
    if (sourceImage && mapArea) {
      const img = new Image()
      img.onload = () => {
        // Create canvas for the cropped map area
        const canvas = document.createElement('canvas')
        const ctx = canvas.getContext('2d')
        
        if (!ctx) return

        canvas.width = mapArea.width
        canvas.height = mapArea.height

        // Draw the cropped map area
        ctx.drawImage(
          img,
          mapArea.x, mapArea.y, mapArea.width, mapArea.height, // Source rectangle
          0, 0, mapArea.width, mapArea.height // Destination rectangle
        )

        // Convert to blob URL for display
        canvas.toBlob((blob) => {
          if (blob) {
            const url = URL.createObjectURL(blob)
            setCroppedMapUrl(url)
          }
        })
      }
      
      img.src = URL.createObjectURL(sourceImage)
      
      return () => {
        if (croppedMapUrl) {
          URL.revokeObjectURL(croppedMapUrl)
        }
      }
    } else {
      setCroppedMapUrl(null)
      setStartPoint(null)
      setEndPoint(null)
      onPointsChange(null, null)
    }
  }, [sourceImage, mapArea, onPointsChange])

  // Reset points when cropped map URL changes
  useEffect(() => {
    setStartPoint(null)
    setEndPoint(null)
    onPointsChange(null, null)
  }, [croppedMapUrl, onPointsChange])

  const handleImageClick = useCallback((e: React.MouseEvent<HTMLImageElement>) => {
    if (!imgRef.current) return

    const rect = imgRef.current.getBoundingClientRect()
    const scaleX = imgRef.current.naturalWidth / rect.width
    const scaleY = imgRef.current.naturalHeight / rect.height

    const point: Point = {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    }

    if (!startPoint) {
      setStartPoint(point)
      onPointsChange(point, null)
    } else if (!endPoint) {
      setEndPoint(point)
      onPointsChange(startPoint, point)
    } else {
      // Reset and start over
      setStartPoint(point)
      setEndPoint(null)
      onPointsChange(point, null)
    }
  }, [startPoint, endPoint, onPointsChange])

  const handleClear = useCallback(() => {
    setStartPoint(null)
    setEndPoint(null)
    onPointsChange(null, null)
  }, [onPointsChange])

  if (!sourceImage || !mapArea) {
    return (
      <div className="text-center text-muted-foreground p-8 border-2 border-dashed border-border rounded-lg">
        <p>Please select the map area first</p>
      </div>
    )
  }

  if (!croppedMapUrl) {
    return (
      <div className="text-center text-muted-foreground p-8 border-2 border-dashed border-border rounded-lg">
        <p>Loading map area...</p>
      </div>
    )
  }

  return (
    <div className={cn("space-y-4", className)}>
      <div className="relative inline-block">
        <img
          ref={imgRef}
          src={croppedMapUrl}
          alt="Map area for cross-section selection"
          className="max-w-full h-auto border border-border rounded-lg cursor-crosshair"
          onClick={handleImageClick}
        />
        
        {/* Overlay canvas for drawing points and line */}
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 pointer-events-none"
          style={{
            width: '100%',
            height: '100%',
          }}
        />
        
        {/* Draw points and line on the image */}
        <svg
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
          viewBox={`0 0 ${imgRef.current?.naturalWidth || 1} ${imgRef.current?.naturalHeight || 1}`}
          preserveAspectRatio="none"
        >
          {startPoint && (
            <circle
              cx={startPoint.x}
              cy={startPoint.y}
              r={5}
              fill="#3B82F6"
              stroke="white"
              strokeWidth={2}
            />
          )}
          {endPoint && (
            <circle
              cx={endPoint.x}
              cy={endPoint.y}
              r={5}
              fill="#EF4444"
              stroke="white"
              strokeWidth={2}
            />
          )}
          {startPoint && endPoint && (
            <line
              x1={startPoint.x}
              y1={startPoint.y}
              x2={endPoint.x}
              y2={endPoint.y}
              stroke="#10B981"
              strokeWidth={3}
              strokeDasharray="5,5"
            />
          )}
        </svg>
      </div>
      
      <div className="flex items-center justify-between">
        <div className="text-sm text-muted-foreground">
          {!startPoint && "Click to select start point"}
          {startPoint && !endPoint && "Click to select end point"}
          {startPoint && endPoint && "Cross-section line selected"}
        </div>
        
        {(startPoint || endPoint) && (
          <button
            type="button"
            onClick={handleClear}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            Clear points
          </button>
        )}
      </div>
      
      {startPoint && endPoint && (
        <div className="text-xs text-muted-foreground">
          <div>Start: ({Math.round(startPoint.x)}, {Math.round(startPoint.y)})</div>
          <div>End: ({Math.round(endPoint.x)}, {Math.round(endPoint.y)})</div>
        </div>
      )}
    </div>
  )
}