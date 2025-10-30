"use client"

import { AlertTriangle } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { useRouter } from "next/navigation"

interface UnauthorizedErrorProps {
  message?: string
  onClose?: () => void
}

export function UnauthorizedError({ 
  message = "Your session has expired. Please log in again.", 
  onClose 
}: UnauthorizedErrorProps) {
  const router = useRouter()

  const handleLoginRedirect = () => {
    onClose?.()
    router.push("/login")
  }

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
      <div className="relative z-10 w-full max-w-md">
        <Alert variant="destructive" className="border-red-500 bg-red-50 dark:bg-red-950/50">
          <AlertTriangle className="h-5 w-5 text-red-600" />
          <AlertDescription className="text-red-800 dark:text-red-200">
            {message}
          </AlertDescription>
        </Alert>
        <div className="mt-4 flex justify-center gap-2">
          <Button 
            variant="default" 
            onClick={handleLoginRedirect}
            className="bg-red-600 hover:bg-red-700 text-white"
          >
            Go to Login
          </Button>
          {onClose && (
            <Button variant="outline" onClick={onClose}>
              Close
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}