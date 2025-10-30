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
  const imgRef = useRef<HTMLImageElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const onPointsChangeRef = useRef(onPointsChange)
  
  // Update ref when callback changes
  useEffect(() => {
    onPointsChangeRef.current = onPointsChange
  }, [onPointsChange])

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
      // Only clear points if we don't have a map area at all
      if (!mapArea) {
        setStartPoint(null)
        setEndPoint(null)
        onPointsChangeRef.current(null, null)
      }
    }
  }, [sourceImage, mapArea])

  const clearPoints = useCallback(() => {
    setStartPoint(null)
    setEndPoint(null)
    onPointsChangeRef.current(null, null)
  }, [])

  const getRelativeCoordinates = useCallback((e: React.MouseEvent<HTMLImageElement>): Point => {
    if (!imgRef.current) return { x: 0, y: 0 }

    const rect = imgRef.current.getBoundingClientRect()
    const scaleX = imgRef.current.naturalWidth / rect.width
    const scaleY = imgRef.current.naturalHeight / rect.height

    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    }
  }, [])

  const handleImageClick = useCallback((e: React.MouseEvent<HTMLImageElement>) => {
    e.preventDefault()
    const point = getRelativeCoordinates(e)
    console.log('[MapPointSelector] Clicked at:', point)

    if (!startPoint) {
      // Set start point
      console.log('[MapPointSelector] Setting start point:', point)
      setStartPoint(point)
      onPointsChangeRef.current(point, null)
    } else if (!endPoint) {
      // Set end point
      console.log('[MapPointSelector] Setting end point:', point)
      setEndPoint(point)
      onPointsChangeRef.current(startPoint, point)
    } else {
      // Reset and start over with new start point
      console.log('[MapPointSelector] Resetting with new start point:', point)
      setStartPoint(point)
      setEndPoint(null)
      onPointsChangeRef.current(point, null)
    }
  }, [startPoint, endPoint, onPointsChange, getRelativeCoordinates])

  const handleClear = useCallback(() => {
    clearPoints()
  }, [clearPoints])

  if (!sourceImage || !mapArea) {
    return (
      <div className={cn("text-center text-muted-foreground p-8 border-2 border-dashed border-border rounded-lg", className)}>
        <p>Please select the map area first</p>
      </div>
    )
  }

  if (!croppedMapUrl) {
    return (
      <div className={cn("text-center text-muted-foreground p-8 border-2 border-dashed border-border rounded-lg", className)}>
        <p>Loading map area...</p>
      </div>
    )
  }

  return (
    <div className={cn("space-y-4", className)}>
      <div 
        ref={containerRef}
        className="relative inline-block border border-border rounded-lg overflow-hidden bg-white"
      >
        <img
          ref={imgRef}
          src={croppedMapUrl}
          alt="Map area for cross-section selection"
          className="max-w-full h-auto cursor-crosshair block"
          onClick={handleImageClick}
          draggable={false}
        />
        
        {/* SVG overlay for points and line */}
        {imgRef.current && (
          <svg
            className="absolute top-0 left-0 w-full h-full pointer-events-none"
            style={{
              width: imgRef.current.offsetWidth,
              height: imgRef.current.offsetHeight
            }}
            viewBox={`0 0 ${imgRef.current.naturalWidth} ${imgRef.current.naturalHeight}`}
            preserveAspectRatio="none"
          >
            {/* Start point */}
            {startPoint && (
              <g>
                <circle
                  cx={startPoint.x}
                  cy={startPoint.y}
                  r={8}
                  fill="#3B82F6"
                  stroke="white"
                  strokeWidth={3}
                />
                <text
                  x={startPoint.x}
                  y={startPoint.y - 15}
                  textAnchor="middle"
                  className="fill-blue-600 text-sm font-bold"
                  style={{ fontSize: '14px' }}
                >
                  START
                </text>
              </g>
            )}
            
            {/* End point */}
            {endPoint && (
              <g>
                <circle
                  cx={endPoint.x}
                  cy={endPoint.y}
                  r={8}
                  fill="#EF4444"
                  stroke="white"
                  strokeWidth={3}
                />
                <text
                  x={endPoint.x}
                  y={endPoint.y - 15}
                  textAnchor="middle"
                  className="fill-red-600 text-sm font-bold"
                  style={{ fontSize: '14px' }}
                >
                  END
                </text>
              </g>
            )}
            
            {/* Connecting line */}
            {startPoint && endPoint && (
              <line
                x1={startPoint.x}
                y1={startPoint.y}
                x2={endPoint.x}
                y2={endPoint.y}
                stroke="#10B981"
                strokeWidth={4}
                strokeDasharray="8,4"
                opacity={0.8}
              />
            )}
          </svg>
        )}
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
            className="px-3 py-1 text-xs text-muted-foreground hover:text-foreground transition-colors bg-muted hover:bg-muted/80 rounded"
          >
            Clear Points
          </button>
        )}
      </div>
      
      {/* Point coordinates display */}
      {(startPoint || endPoint) && (
        <div className="space-y-1">
          {startPoint && (
            <div className="flex items-center gap-2 text-xs">
              <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
              <span className="font-medium">Start:</span>
              <span className="text-muted-foreground">
                ({Math.round(startPoint.x)}, {Math.round(startPoint.y)})
              </span>
            </div>
          )}
          {endPoint && (
            <div className="flex items-center gap-2 text-xs">
              <div className="w-3 h-3 bg-red-500 rounded-full"></div>
              <span className="font-medium">End:</span>
              <span className="text-muted-foreground">
                ({Math.round(endPoint.x)}, {Math.round(endPoint.y)})
              </span>
            </div>
          )}
        </div>
      )}

      {/* Instructions */}
      <div className="text-xs text-muted-foreground p-2 bg-blue-50 rounded">
        <p><strong>Instructions:</strong></p>
        <ul className="mt-1 space-y-1 list-disc list-inside">
          <li>Click on the map to place the start point (blue)</li>
          <li>Click again to place the end point (red)</li>
          <li>Click a third time to restart with a new start point</li>
        </ul>
      </div>
    </div>
  )
}