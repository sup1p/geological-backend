// Test file to verify the new single image upload workflow
// This is not a formal test - just a demonstration of the new flow

/*
NEW WORKFLOW OVERVIEW:

1. User uploads ONE image file (instead of three separate images)
2. User selects THREE areas on that image:
   - Map area (geological map)
   - Legend area (map legend) 
   - Column area (stratigraphic column)
3. User selects two points on the MAP area to define cross-section line
4. System crops the three areas from the source image
5. System submits the cropped images + cross-section coordinates to backend

KEY COMPONENTS:
- ImageAreaSelector: Allows user to draw rectangles on source image to select map/legend/column areas
- MapPointSelector: Shows only the cropped map area and allows selecting start/end points for cross-section
- Image cropping utilities: Extract the selected areas from source image as separate image files
- Updated upload modal: Orchestrates the entire workflow

USER EXPERIENCE IMPROVEMENTS:
- Single file upload instead of managing three separate files
- Visual selection of areas instead of guessing which part is map/legend/column
- Immediate preview of selected areas
- Cross-section line selection directly on the relevant map area
- Clear validation feedback throughout the process

TECHNICAL IMPLEMENTATION:
- Canvas API for drawing selection rectangles and points
- Image cropping using HTML5 Canvas
- File blob creation for uploading cropped areas
- Coordinate mapping between full image and cropped map area
- Maintains backward compatibility with existing backend API
*/

export const newWorkflowDemo = {
  title: "Single Image Upload with Area Selection",
  description: "User uploads one image and selects areas interactively",
  
  steps: [
    "1. User clicks upload and selects geological image file",
    "2. ImageAreaSelector shows the image with drawing tools",
    "3. User draws rectangle around map area (labeled 'Map')",
    "4. User draws rectangle around legend area (labeled 'Legend')", 
    "5. User draws rectangle around column area (labeled 'Column')",
    "6. MapPointSelector crops and shows only the map area",
    "7. User clicks two points to define cross-section line",
    "8. System validates all selections are complete",
    "9. System crops three areas into separate image files",
    "10. System uploads cropped images + coordinates to backend"
  ],
  
  benefits: [
    "Simplified UX - only one file to manage",
    "Visual selection reduces user confusion", 
    "Immediate preview of selections",
    "Cross-section selection on relevant area only",
    "Better error handling and validation"
  ]
}