"use client"

import React, { useState, useRef, useCallback } from "react"
import { cn } from "@/lib/utils"

interface Point {
  x: number
  y: number
}

interface ImagePointSelectorProps {
  imageFile: File | null
  onPointsChange: (startPoint: Point | null, endPoint: Point | null) => void
  className?: string
}

export function ImagePointSelector({ imageFile, onPointsChange, className }: ImagePointSelectorProps) {
  const [startPoint, setStartPoint] = useState<Point | null>(null)
  const [endPoint, setEndPoint] = useState<Point | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number } | null>(null)
  const [imageScale, setImageScale] = useState<number>(1)
  const imgRef = useRef<HTMLImageElement>(null)

  // Create object URL when image file changes
  React.useEffect(() => {
    if (imageFile) {
      const url = URL.createObjectURL(imageFile)
      setImageUrl(url)
      
      return () => {
        URL.revokeObjectURL(url)
      }
    } else {
      setImageUrl(null)
      setStartPoint(null)
      setEndPoint(null)
      setImageDimensions(null)
    }
  }, [imageFile])

  const handleImageLoad = useCallback(() => {
    if (imgRef.current) {
      setImageDimensions({
        width: imgRef.current.naturalWidth,
        height: imgRef.current.naturalHeight
      })
    }
  }, [])

  const handleImageClick = useCallback((e: React.MouseEvent<HTMLImageElement>) => {
    if (!imgRef.current || !imageDimensions) return

    const rect = imgRef.current.getBoundingClientRect()
    const scaleX = imageDimensions.width / rect.width
    const scaleY = imageDimensions.height / rect.height
    
    const x = Math.round((e.clientX - rect.left) * scaleX)
    const y = Math.round((e.clientY - rect.top) * scaleY)

    if (!startPoint) {
      const newStartPoint = { x, y }
      setStartPoint(newStartPoint)
      onPointsChange(newStartPoint, endPoint)
    } else if (!endPoint) {
      const newEndPoint = { x, y }
      setEndPoint(newEndPoint)
      onPointsChange(startPoint, newEndPoint)
    } else {
      // Reset and start over
      const newStartPoint = { x, y }
      setStartPoint(newStartPoint)
      setEndPoint(null)
      onPointsChange(newStartPoint, null)
    }
  }, [startPoint, endPoint, imageDimensions, onPointsChange])

  const getPointScreenPosition = (point: Point) => {
    if (!imgRef.current || !imageDimensions) return null

    const rect = imgRef.current.getBoundingClientRect()
    const scaleX = rect.width / imageDimensions.width
    const scaleY = rect.height / imageDimensions.height

    return {
      x: point.x * scaleX,
      y: point.y * scaleY
    }
  }

  const clearPoints = () => {
    setStartPoint(null)
    setEndPoint(null)
    onPointsChange(null, null)
  }

  if (!imageUrl) {
    return (
      <div className={cn("flex items-center justify-center h-64 border-2 border-dashed border-border rounded-lg", className)}>
        <p className="text-muted-foreground">Upload a geological map to select cross-section points</p>
      </div>
    )
  }

  return (
    <div className={cn("space-y-4", className)}>
      <div className="flex justify-between items-center">
        <div className="text-sm text-muted-foreground">
          {!startPoint && "Click to select the start point"}
          {startPoint && !endPoint && "Click to select the end point"}
          {startPoint && endPoint && "Both points selected. Click to start over."}
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <button
              type="button"
              onClick={() => setImageScale(Math.max(0.5, imageScale - 0.25))}
              className="px-2 py-1 bg-muted rounded hover:bg-muted/80"
              disabled={imageScale <= 0.5}
            >
              -
            </button>
            <span className="w-12 text-center">{Math.round(imageScale * 100)}%</span>
            <button
              type="button"
              onClick={() => setImageScale(Math.min(3, imageScale + 0.25))}
              className="px-2 py-1 bg-muted rounded hover:bg-muted/80"
              disabled={imageScale >= 3}
            >
              +
            </button>
          </div>
          {(startPoint || endPoint) && (
            <button
              type="button"
              onClick={clearPoints}
              className="text-xs text-muted-foreground hover:text-foreground underline"
            >
              Clear points
            </button>
          )}
        </div>
      </div>

      <div className="relative border rounded-lg bg-muted" style={{ maxHeight: '70vh', overflow: 'auto' }}>
        <img
          ref={imgRef}
          src={imageUrl}
          alt="Geological map for point selection"
          className="cursor-crosshair block"
          style={{ 
            width: `${imageScale * 100}%`,
            height: 'auto',
            minWidth: '100%'
          }}
          onClick={handleImageClick}
          onLoad={handleImageLoad}
        />
        
        {/* Start point marker */}
        {startPoint && (() => {
          const screenPos = getPointScreenPosition(startPoint)
          return screenPos ? (
            <div
              className="absolute w-4 h-4 bg-green-500 border-2 border-white rounded-full shadow-lg transform -translate-x-1/2 -translate-y-1/2 pointer-events-none z-10"
              style={{ left: screenPos.x, top: screenPos.y }}
            >
              <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 text-xs bg-green-500 text-white px-2 py-1 rounded whitespace-nowrap font-medium">
                Start ({startPoint.x}, {startPoint.y})
              </div>
            </div>
          ) : null
        })()}

        {/* End point marker */}
        {endPoint && (() => {
          const screenPos = getPointScreenPosition(endPoint)
          return screenPos ? (
            <div
              className="absolute w-4 h-4 bg-red-500 border-2 border-white rounded-full shadow-lg transform -translate-x-1/2 -translate-y-1/2 pointer-events-none z-10"
              style={{ left: screenPos.x, top: screenPos.y }}
            >
              <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 text-xs bg-red-500 text-white px-2 py-1 rounded whitespace-nowrap font-medium">
                End ({endPoint.x}, {endPoint.y})
              </div>
            </div>
          ) : null
        })()}

        {/* Line between points */}
        {startPoint && endPoint && (() => {
          const startScreenPos = getPointScreenPosition(startPoint)
          const endScreenPos = getPointScreenPosition(endPoint)
          if (!startScreenPos || !endScreenPos) return null

          const length = Math.sqrt(
            Math.pow(endScreenPos.x - startScreenPos.x, 2) + 
            Math.pow(endScreenPos.y - startScreenPos.y, 2)
          )
          const angle = Math.atan2(
            endScreenPos.y - startScreenPos.y,
            endScreenPos.x - startScreenPos.x
          ) * 180 / Math.PI

          return (
            <div
              className="absolute h-1 bg-blue-500 shadow-lg pointer-events-none origin-left z-5"
              style={{
                left: startScreenPos.x,
                top: startScreenPos.y,
                width: length,
                transform: `rotate(${angle}deg)`
              }}
            />
          )
        })()}
      </div>

      {startPoint && endPoint && (
        <div className="text-xs text-muted-foreground bg-muted p-2 rounded">
          <div>Start Point: ({startPoint.x}, {startPoint.y})</div>
          <div>End Point: ({endPoint.x}, {endPoint.y})</div>
          <div>
            Distance: {Math.round(Math.sqrt(
              Math.pow(endPoint.x - startPoint.x, 2) + 
              Math.pow(endPoint.y - startPoint.y, 2)
            ))} pixels
          </div>
        </div>
      )}
    </div>
  )
}