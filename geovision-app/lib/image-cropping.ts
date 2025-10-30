// Utility functions for cropping images based on selected areas

interface Area {
  x: number
  y: number
  width: number
  height: number
  label: string
}

// Convert canvas area to cropped image blob
export async function cropImageArea(
  imageFile: File,
  area: Area,
  outputFormat: string = 'image/png'
): Promise<File> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      try {
        // Create canvas for cropping
        const canvas = document.createElement('canvas')
        const ctx = canvas.getContext('2d')
        
        if (!ctx) {
          reject(new Error('Could not get canvas context'))
          return
        }

        // Set canvas size to crop area
        canvas.width = area.width
        canvas.height = area.height

        // Draw the cropped portion
        ctx.drawImage(
          img,
          area.x, area.y, area.width, area.height, // Source rectangle
          0, 0, area.width, area.height // Destination rectangle
        )

        // Convert to blob
        canvas.toBlob((blob) => {
          if (!blob) {
            reject(new Error('Failed to create blob from canvas'))
            return
          }

          // Create File from blob
          const croppedFile = new File(
            [blob],
            `${area.label.toLowerCase().replace(/\s+/g, '_')}_${Date.now()}.png`,
            { type: outputFormat }
          )

          resolve(croppedFile)
        }, outputFormat)
      } catch (error) {
        reject(error)
      }
    }

    img.onerror = () => {
      reject(new Error('Failed to load image'))
    }

    img.src = URL.createObjectURL(imageFile)
  })
}

// Crop all three areas from a single image
export async function cropAllAreas(
  imageFile: File,
  areas: {
    map: Area | null
    legend: Area | null
    column: Area | null
  }
): Promise<{
  mapImage: File | null
  legendImage: File | null
  columnImage: File | null
}> {
  const results = {
    mapImage: null as File | null,
    legendImage: null as File | null,
    columnImage: null as File | null
  }

  try {
    // Crop map area
    if (areas.map) {
      results.mapImage = await cropImageArea(imageFile, areas.map)
    }

    // Crop legend area
    if (areas.legend) {
      results.legendImage = await cropImageArea(imageFile, areas.legend)
    }

    // Crop column area
    if (areas.column) {
      results.columnImage = await cropImageArea(imageFile, areas.column)
    }

    return results
  } catch (error) {
    console.error('Error cropping areas:', error)
    throw error
  }
}

// Validate that all required areas are selected
export function validateAreas(areas: {
  map: Area | null
  legend: Area | null
  column: Area | null
}): { isValid: boolean; missingAreas: string[] } {
  const missingAreas: string[] = []

  if (!areas.map) missingAreas.push('Geological Map')
  if (!areas.legend) missingAreas.push('Legend')
  if (!areas.column) missingAreas.push('Stratigraphic Column')

  return {
    isValid: missingAreas.length === 0,
    missingAreas
  }
}

// Check if area is too small (minimum size validation)
export function validateAreaSize(area: Area, minWidth = 50, minHeight = 50): boolean {
  return area.width >= minWidth && area.height >= minHeight
}

// Preview cropped area as data URL for debugging
export async function previewCroppedArea(
  imageFile: File,
  area: Area
): Promise<string> {
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  
  if (!ctx) {
    throw new Error('Could not get canvas context')
  }

  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      try {
        canvas.width = area.width
        canvas.height = area.height
        
        ctx.drawImage(
          img,
          area.x, area.y, area.width, area.height,
          0, 0, area.width, area.height
        )
        
        resolve(canvas.toDataURL())
      } catch (error) {
        reject(error)
      }
    }
    
    img.onerror = () => reject(new Error('Failed to load image'))
    img.src = URL.createObjectURL(imageFile)
  })
}