"use client"

import React, { useState, useRef, useCallback, useEffect } from "react"
import { cn } from "@/lib/utils"
import { Check, X, ZoomIn, ZoomOut } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"

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

interface ImageAreaSelectorProps {
  imageFile: File | null
  onAreasChange: (areas: { map: Area | null; legend: Area | null; column: Area | null }) => void
  className?: string
}

type SelectionMode = 'map' | 'legend' | 'column' | null

const AREA_COLORS = {
  map: '#3B82F6',      // Blue
  legend: '#EF4444',   // Red  
  column: '#10B981'    // Green
}

const AREA_LABELS = {
  map: 'Map Area',
  legend: 'Legend Area',
  column: 'Column Area'
}

export function ImageAreaSelector({ imageFile, onAreasChange, className }: ImageAreaSelectorProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  
  const [image, setImage] = useState<HTMLImageElement | null>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [startPoint, setStartPoint] = useState<Point | null>(null)
  const [currentRect, setCurrentRect] = useState<Area | null>(null)
  const [selectionMode, setSelectionMode] = useState<SelectionMode>(null)
  const [scale, setScale] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  
  const [areas, setAreas] = useState<{
    map: Area | null
    legend: Area | null
    column: Area | null
  }>({
    map: null,
    legend: null,
    column: null
  })

  // Auto-scroll parameters
  const SCROLL_MARGIN = 50
  const SCROLL_SPEED = 5
  const scrollTimerRef = useRef<NodeJS.Timeout | null>(null)

  // Load image and setup canvas
  useEffect(() => {
    if (!imageFile) {
      setImage(null)
      resetAreas()
      return
    }

    const img = new Image()
    img.onload = () => {
      setImage(img)
      setupCanvas(img)
    }
    img.src = URL.createObjectURL(imageFile)

    return () => {
      if (img.src) URL.revokeObjectURL(img.src)
    }
  }, [imageFile])

  const resetAreas = useCallback(() => {
    const emptyAreas = { map: null, legend: null, column: null }
    setAreas(emptyAreas)
    onAreasChange(emptyAreas)
    setSelectionMode(null)
  }, [onAreasChange])

  const setupCanvas = useCallback((img: HTMLImageElement) => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    // Calculate scale to fit image in container while maintaining aspect ratio
    const containerWidth = container.clientWidth - 32 // Account for padding
    const containerHeight = Math.min(600, window.innerHeight * 0.6) // Max height
    
    const scaleX = containerWidth / img.naturalWidth
    const scaleY = containerHeight / img.naturalHeight
    const newScale = Math.min(scaleX, scaleY, 1) // Don't upscale
    
    setScale(newScale)
    
    canvas.width = img.naturalWidth * newScale
    canvas.height = img.naturalHeight * newScale
    
    drawCanvas()
  }, [])

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx || !image) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw image
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height)

    // Draw existing areas
    Object.entries(areas).forEach(([type, area]) => {
      if (area) {
        drawArea(ctx, area, AREA_COLORS[type as keyof typeof AREA_COLORS], true)
      }
    })

    // Draw current selection
    if (currentRect && selectionMode) {
      drawArea(ctx, currentRect, AREA_COLORS[selectionMode], false)
    }
  }, [image, areas, currentRect, selectionMode])

  const drawArea = useCallback((
    ctx: CanvasRenderingContext2D, 
    area: Area, 
    color: string, 
    isComplete: boolean
  ) => {
    const scaledArea = {
      x: area.x * scale,
      y: area.y * scale,
      width: area.width * scale,
      height: area.height * scale
    }

    // Draw rectangle
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.setLineDash(isComplete ? [] : [5, 5])
    ctx.strokeRect(scaledArea.x, scaledArea.y, scaledArea.width, scaledArea.height)

    // Fill with semi-transparent color
    ctx.fillStyle = color + '20'
    ctx.fillRect(scaledArea.x, scaledArea.y, scaledArea.width, scaledArea.height)

    // Draw label
    ctx.fillStyle = color
    ctx.font = '14px sans-serif'
    ctx.fillText(
      area.label, 
      scaledArea.x + 5, 
      scaledArea.y + 20
    )

    // Draw checkmark if complete
    if (isComplete) {
      ctx.fillStyle = '#ffffff'
      ctx.fillRect(scaledArea.x + scaledArea.width - 25, scaledArea.y + 5, 20, 20)
      ctx.strokeStyle = color
      ctx.strokeRect(scaledArea.x + scaledArea.width - 25, scaledArea.y + 5, 20, 20)
      
      // Draw checkmark
      ctx.strokeStyle = '#10B981'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(scaledArea.x + scaledArea.width - 20, scaledArea.y + 12)
      ctx.lineTo(scaledArea.x + scaledArea.width - 15, scaledArea.y + 17)
      ctx.lineTo(scaledArea.x + scaledArea.width - 10, scaledArea.y + 10)
      ctx.stroke()
    }
  }, [scale])

  // Redraw canvas when areas change
  useEffect(() => {
    drawCanvas()
  }, [drawCanvas])

  const getMousePos = useCallback((e: React.MouseEvent<HTMLCanvasElement>): Point => {
    const canvas = canvasRef.current
    if (!canvas) return { x: 0, y: 0 }

    const rect = canvas.getBoundingClientRect()
    return {
      x: ((e.clientX - rect.left) / scale),
      y: ((e.clientY - rect.top) / scale)
    }
  }, [scale])

  const startAutoScroll = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const container = containerRef.current
    if (!container) return

    // Stop any existing scroll
    stopAutoScroll()

    const containerRect = container.getBoundingClientRect()
    const mouseX = e.clientX - containerRect.left
    const mouseY = e.clientY - containerRect.top

    let scrollX = 0
    let scrollY = 0

    // Check if near edges and need to scroll
    if (mouseX < SCROLL_MARGIN && container.scrollLeft > 0) scrollX = -SCROLL_SPEED
    if (mouseX > containerRect.width - SCROLL_MARGIN && container.scrollLeft < container.scrollWidth - container.clientWidth) scrollX = SCROLL_SPEED
    if (mouseY < SCROLL_MARGIN && container.scrollTop > 0) scrollY = -SCROLL_SPEED  
    if (mouseY > containerRect.height - SCROLL_MARGIN && container.scrollTop < container.scrollHeight - container.clientHeight) scrollY = SCROLL_SPEED

    if (scrollX !== 0 || scrollY !== 0) {
      scrollTimerRef.current = setInterval(() => {
        container.scrollLeft += scrollX
        container.scrollTop += scrollY
      }, 16)
    }
  }, [])

  const stopAutoScroll = useCallback(() => {
    if (scrollTimerRef.current) {
      clearInterval(scrollTimerRef.current)
      scrollTimerRef.current = null
    }
  }, [])

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!selectionMode) return

    const pos = getMousePos(e)
    setStartPoint(pos)
    setIsDrawing(true)
    setCurrentRect({
      x: pos.x,
      y: pos.y,
      width: 0,
      height: 0,
      label: AREA_LABELS[selectionMode]
    })

    startAutoScroll(e)
  }, [selectionMode, getMousePos, startAutoScroll])

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !startPoint || !selectionMode) return

    const pos = getMousePos(e)
    const width = pos.x - startPoint.x
    const height = pos.y - startPoint.y

    setCurrentRect({
      x: width > 0 ? startPoint.x : pos.x,
      y: height > 0 ? startPoint.y : pos.y,
      width: Math.abs(width),
      height: Math.abs(height),
      label: AREA_LABELS[selectionMode]
    })

    startAutoScroll(e)
  }, [isDrawing, startPoint, selectionMode, getMousePos, startAutoScroll])

  const handleMouseUp = useCallback(() => {
    if (!isDrawing || !currentRect || !selectionMode) return

    stopAutoScroll()
    setIsDrawing(false)
    setStartPoint(null)

    // Minimum area size validation
    if (currentRect.width > 10 && currentRect.height > 10) {
      const newAreas = {
        ...areas,
        [selectionMode]: currentRect
      }
      setAreas(newAreas)
      onAreasChange(newAreas)
    }

    setCurrentRect(null)
    setSelectionMode(null)
  }, [isDrawing, currentRect, selectionMode, areas, onAreasChange, stopAutoScroll])

  const clearArea = useCallback((areaType: keyof typeof areas) => {
    const newAreas = {
      ...areas,
      [areaType]: null
    }
    setAreas(newAreas)
    onAreasChange(newAreas)
  }, [areas, onAreasChange])

  const getCompletionStats = useCallback(() => {
    const completed = Object.values(areas).filter(area => area !== null).length
    return { completed, total: 3 }
  }, [areas])

  if (!imageFile) {
    return (
      <div className={cn("text-center text-muted-foreground p-8 border-2 border-dashed border-border rounded-lg", className)}>
        <p>Please upload an image first</p>
      </div>
    )
  }

  const { completed, total } = getCompletionStats()

  return (
    <div className={cn("space-y-4", className)}>
      {/* Progress indicator */}
      <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
        <div className="flex items-center gap-2">
          <div className="text-sm font-medium">
            Areas Selected: {completed}/{total}
          </div>
          {completed === total && (
            <div className="flex items-center gap-1 text-green-600">
              <Check className="h-4 w-4" />
              <span className="text-xs">Complete</span>
            </div>
          )}
        </div>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={resetAreas}
          disabled={completed === 0}
        >
          Clear All
        </Button>
      </div>

      {/* Selection buttons */}
      <div className="grid grid-cols-3 gap-2">
        {Object.entries(AREA_LABELS).map(([type, label]) => {
          const areaType = type as keyof typeof areas
          const isSelected = areas[areaType] !== null
          const isActive = selectionMode === type
          
          return (
            <div key={type} className="space-y-1">
              <Button
                type="button"
                variant={isActive ? "default" : isSelected ? "secondary" : "outline"}
                size="sm"
                className="w-full relative"
                onClick={() => setSelectionMode(isActive ? null : areaType)}
                disabled={isDrawing}
                style={isSelected ? { 
                  borderColor: AREA_COLORS[areaType],
                  backgroundColor: AREA_COLORS[areaType] + '20'
                } : undefined}
              >
                {isSelected && (
                  <Check className="h-3 w-3 mr-1" style={{ color: AREA_COLORS[areaType] }} />
                )}
                {label}
              </Button>
              {isSelected && (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="w-full h-6 text-xs"
                  onClick={() => clearArea(areaType)}
                >
                  <X className="h-3 w-3 mr-1" />
                  Clear
                </Button>
              )}
            </div>
          )
        })}
      </div>

      {/* Instructions */}
      <div className="text-sm text-muted-foreground p-3 bg-blue-50 rounded-lg">
        {selectionMode ? (
          <p>Click and drag to select the <strong>{AREA_LABELS[selectionMode]}</strong></p>
        ) : completed < total ? (
          <p>Click on a button above to start selecting an area</p>
        ) : (
          <p>All areas selected! You can now proceed to the next step.</p>
        )}
      </div>

      {/* Canvas container with scroll */}
      <div 
        ref={containerRef}
        className="border border-border rounded-lg overflow-auto max-h-[600px] bg-gray-50"
        style={{ maxHeight: Math.min(600, window.innerHeight * 0.6) }}
      >
        <canvas
          ref={canvasRef}
          className="cursor-crosshair block"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={stopAutoScroll}
        />
      </div>
      
      {/* Area details */}
      {Object.entries(areas).some(([_, area]) => area !== null) && (
        <div className="space-y-2">
          <Label className="text-xs font-medium text-muted-foreground">Selected Areas:</Label>
          <div className="grid grid-cols-1 gap-2 text-xs">
            {Object.entries(areas).map(([type, area]) => {
              if (!area) return null
              return (
                <div key={type} className="flex items-center justify-between p-2 bg-muted rounded">
                  <span className="font-medium" style={{ color: AREA_COLORS[type as keyof typeof AREA_COLORS] }}>
                    {area.label}
                  </span>
                  <span className="text-muted-foreground">
                    {Math.round(area.width)} × {Math.round(area.height)} px
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}