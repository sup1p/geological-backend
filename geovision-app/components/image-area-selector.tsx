"use client"

import { useState, useRef, useCallback, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"

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
}

type SelectionMode = 'map' | 'legend' | 'column' | null

export function ImageAreaSelector({ imageFile, onAreasChange }: ImageAreaSelectorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [image, setImage] = useState<HTMLImageElement | null>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [startPoint, setStartPoint] = useState<{ x: number; y: number } | null>(null)
  const [currentArea, setCurrentArea] = useState<Area | null>(null)
  const [selectionMode, setSelectionMode] = useState<SelectionMode>(null)
  const [areas, setAreas] = useState<{
    map: Area | null
    legend: Area | null
    column: Area | null
  }>({
    map: null,
    legend: null,
    column: null
  })

  // Load image when file changes
  useEffect(() => {
    if (!imageFile) {
      setImage(null)
      setAreas({ map: null, legend: null, column: null })
      onAreasChange({ map: null, legend: null, column: null })
      return
    }

    const img = new Image()
    img.onload = () => {
      setImage(img)
      // Reset areas when new image is loaded
      setAreas({ map: null, legend: null, column: null })
      onAreasChange({ map: null, legend: null, column: null })
    }
    img.src = URL.createObjectURL(imageFile)

    return () => {
      URL.revokeObjectURL(img.src)
    }
  }, [imageFile, onAreasChange])

  // Draw on canvas when image or areas change
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !image) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size to match image
    canvas.width = image.width
    canvas.height = image.height

    // Draw image
    ctx.drawImage(image, 0, 0)

    // Draw existing areas
    Object.entries(areas).forEach(([type, area]) => {
      if (area) {
        drawArea(ctx, area, getAreaColor(type as keyof typeof areas))
      }
    })

    // Draw current drawing area
    if (currentArea) {
      drawArea(ctx, currentArea, getAreaColor(selectionMode!), true)
    }
  }, [image, areas, currentArea, selectionMode])

  const getAreaColor = (type: keyof typeof areas) => {
    switch (type) {
      case 'map': return '#3B82F6' // blue
      case 'legend': return '#EF4444' // red
      case 'column': return '#10B981' // green
      default: return '#6B7280' // gray
    }
  }

  const drawArea = (ctx: CanvasRenderingContext2D, area: Area, color: string, isDashed = false) => {
    ctx.save()
    
    // Set line style
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    if (isDashed) {
      ctx.setLineDash([5, 5])
    } else {
      ctx.setLineDash([])
    }
    
    // Draw rectangle
    ctx.strokeRect(area.x, area.y, area.width, area.height)
    
    // Draw semi-transparent fill
    ctx.fillStyle = color + '20'
    ctx.fillRect(area.x, area.y, area.width, area.height)
    
    // Draw label
    ctx.fillStyle = color
    ctx.font = '14px Arial'
    ctx.fillText(area.label, area.x + 5, area.y - 5)
    
    ctx.restore()
  }

  const getCanvasCoordinates = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return { x: 0, y: 0 }

    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height

    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY
    }
  }, [])

  const handleMouseDown = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!selectionMode) return

    const point = getCanvasCoordinates(event)
    setStartPoint(point)
    setIsDrawing(true)
  }, [selectionMode, getCanvasCoordinates])

  const handleMouseMove = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !startPoint || !selectionMode) return

    const point = getCanvasCoordinates(event)
    const area: Area = {
      x: Math.min(startPoint.x, point.x),
      y: Math.min(startPoint.y, point.y),
      width: Math.abs(point.x - startPoint.x),
      height: Math.abs(point.y - startPoint.y),
      label: getAreaLabel(selectionMode)
    }

    setCurrentArea(area)
  }, [isDrawing, startPoint, selectionMode, getCanvasCoordinates])

  const handleMouseUp = useCallback(() => {
    if (!isDrawing || !currentArea || !selectionMode) return

    // Save the area
    const newAreas = { ...areas, [selectionMode]: currentArea }
    setAreas(newAreas)
    onAreasChange(newAreas)

    // Reset drawing state
    setIsDrawing(false)
    setStartPoint(null)
    setCurrentArea(null)
    setSelectionMode(null)
  }, [isDrawing, currentArea, selectionMode, areas, onAreasChange])

  const getAreaLabel = (type: SelectionMode): string => {
    switch (type) {
      case 'map': return 'Geological Map'
      case 'legend': return 'Legend'
      case 'column': return 'Stratigraphic Column'
      default: return ''
    }
  }

  const clearArea = (type: keyof typeof areas) => {
    const newAreas = { ...areas, [type]: null }
    setAreas(newAreas)
    onAreasChange(newAreas)
  }

  if (!imageFile || !image) {
    return (
      <div className="text-center text-muted-foreground">
        Please upload an image first
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-2">
        <Button
          type="button"
          variant={selectionMode === 'map' ? 'default' : 'outline'}
          className="text-blue-600 border-blue-600"
          onClick={() => setSelectionMode('map')}
          disabled={isDrawing}
        >
          {areas.map ? '✓' : ''} Select Map
        </Button>
        <Button
          type="button"
          variant={selectionMode === 'legend' ? 'default' : 'outline'}
          className="text-red-600 border-red-600"
          onClick={() => setSelectionMode('legend')}
          disabled={isDrawing}
        >
          {areas.legend ? '✓' : ''} Select Legend
        </Button>
        <Button
          type="button"
          variant={selectionMode === 'column' ? 'default' : 'outline'}
          className="text-green-600 border-green-600"
          onClick={() => setSelectionMode('column')}
          disabled={isDrawing}
        >
          {areas.column ? '✓' : ''} Select Column
        </Button>
      </div>

      {selectionMode && (
        <div className="text-sm text-muted-foreground text-center">
          Click and drag to select the <span className="font-semibold">{getAreaLabel(selectionMode)}</span> area
        </div>
      )}

      <div className="border rounded-lg p-4 overflow-auto max-h-96">
        <canvas
          ref={canvasRef}
          className="max-w-full h-auto cursor-crosshair"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={() => {
            setIsDrawing(false)
            setStartPoint(null)
            setCurrentArea(null)
          }}
        />
      </div>

      {(areas.map || areas.legend || areas.column) && (
        <div className="space-y-2">
          <Label>Selected Areas:</Label>
          <div className="grid grid-cols-3 gap-2 text-xs">
            {areas.map && (
              <div className="p-2 bg-blue-50 border border-blue-200 rounded">
                <div className="font-semibold text-blue-600">Map</div>
                <div>{Math.round(areas.map.width)}×{Math.round(areas.map.height)}</div>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="mt-1 h-6 text-xs"
                  onClick={() => clearArea('map')}
                >
                  Clear
                </Button>
              </div>
            )}
            {areas.legend && (
              <div className="p-2 bg-red-50 border border-red-200 rounded">
                <div className="font-semibold text-red-600">Legend</div>
                <div>{Math.round(areas.legend.width)}×{Math.round(areas.legend.height)}</div>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="mt-1 h-6 text-xs"
                  onClick={() => clearArea('legend')}
                >
                  Clear
                </Button>
              </div>
            )}
            {areas.column && (
              <div className="p-2 bg-green-50 border border-green-200 rounded">
                <div className="font-semibold text-green-600">Column</div>
                <div>{Math.round(areas.column.width)}×{Math.round(areas.column.height)}</div>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="mt-1 h-6 text-xs"
                  onClick={() => clearArea('column')}
                >
                  Clear
                </Button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}