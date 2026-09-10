param(
    [int]$Count = 100,
    [int]$IntervalSeconds = 3,
    [string]$OutputDir = ".\screenshots",
    [switch]$FullScreen
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class DpiAwareness {
    [DllImport("shcore.dll")]
    public static extern int SetProcessDpiAwareness(int value);

    [DllImport("user32.dll")]
    public static extern bool SetProcessDPIAware();
}
"@

try {
    [void][DpiAwareness]::SetProcessDpiAwareness(2)
}
catch {
    try {
        [void][DpiAwareness]::SetProcessDPIAware()
    }
    catch {
    }
}

$resolvedOutputDir = Join-Path (Get-Location) $OutputDir
if (-not (Test-Path -LiteralPath $resolvedOutputDir)) {
    New-Item -ItemType Directory -Path $resolvedOutputDir | Out-Null
}

function Select-ScreenRegion {
    $virtualBounds = [System.Windows.Forms.SystemInformation]::VirtualScreen
    $form = New-Object System.Windows.Forms.Form
    $form.AutoScaleMode = [System.Windows.Forms.AutoScaleMode]::None
    $form.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::None
    $form.StartPosition = [System.Windows.Forms.FormStartPosition]::Manual
    $form.Bounds = $virtualBounds
    $form.TopMost = $true
    $form.ShowInTaskbar = $false
    $form.KeyPreview = $true
    $form.Cursor = [System.Windows.Forms.Cursors]::Cross
    $form.BackColor = [System.Drawing.Color]::Black
    $form.Opacity = 0.25

    $script:startPoint = $null
    $script:currentPoint = $null
    $script:selectedBounds = $null
    $script:isDragging = $false

    $form.Add_MouseDown({
        if ($_.Button -eq [System.Windows.Forms.MouseButtons]::Left) {
            $script:isDragging = $true
            $script:startPoint = [System.Windows.Forms.Cursor]::Position
            $script:currentPoint = $script:startPoint
            $form.Invalidate()
        }
    })

    $form.Add_MouseMove({
        if ($script:isDragging) {
            $script:currentPoint = [System.Windows.Forms.Cursor]::Position
            $form.Invalidate()
        }
    })

    $form.Add_MouseUp({
        if ($_.Button -eq [System.Windows.Forms.MouseButtons]::Left -and $script:isDragging) {
            $script:isDragging = $false
            $endPoint = [System.Windows.Forms.Cursor]::Position

            $left = [Math]::Min($script:startPoint.X, $endPoint.X)
            $top = [Math]::Min($script:startPoint.Y, $endPoint.Y)
            $width = [Math]::Abs($script:startPoint.X - $endPoint.X)
            $height = [Math]::Abs($script:startPoint.Y - $endPoint.Y)

            if ($width -ge 5 -and $height -ge 5) {
                $script:selectedBounds = New-Object System.Drawing.Rectangle $left, $top, $width, $height
                $form.Close()
            }
        }
    })

    $form.Add_KeyDown({
        if ($_.KeyCode -eq [System.Windows.Forms.Keys]::Escape) {
            $script:selectedBounds = $null
            $form.Close()
        }
    })

    $form.Add_Paint({
        if ($script:startPoint -ne $null -and $script:currentPoint -ne $null) {
            $left = [Math]::Min($script:startPoint.X, $script:currentPoint.X) - $virtualBounds.X
            $top = [Math]::Min($script:startPoint.Y, $script:currentPoint.Y) - $virtualBounds.Y
            $width = [Math]::Abs($script:startPoint.X - $script:currentPoint.X)
            $height = [Math]::Abs($script:startPoint.Y - $script:currentPoint.Y)

            $rect = New-Object System.Drawing.Rectangle $left, $top, $width, $height
            $pen = New-Object System.Drawing.Pen ([System.Drawing.Color]::Red), 3
            $brush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(80, 255, 255, 255))

            try {
                $_.Graphics.FillRectangle($brush, $rect)
                $_.Graphics.DrawRectangle($pen, $rect)
            }
            finally {
                $pen.Dispose()
                $brush.Dispose()
            }
        }
    })

    [void]$form.ShowDialog()
    $form.Dispose()

    return $script:selectedBounds
}

if ($FullScreen) {
    $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
}
else {
    Write-Host "Drag to select the screenshot area. Press Esc to cancel."
    Start-Sleep -Milliseconds 500
    $bounds = Select-ScreenRegion

    if ($null -eq $bounds) {
        Write-Host "Canceled. No screenshots were taken."
        exit 1
    }
}

Write-Host "Saving $Count screenshots to: $resolvedOutputDir"
Write-Host "Interval: $IntervalSeconds seconds"
Write-Host "Capture area: X=$($bounds.X), Y=$($bounds.Y), Width=$($bounds.Width), Height=$($bounds.Height)"
Write-Host "Switch to your game window now. First screenshot starts in 3 seconds..."
Start-Sleep -Seconds 3

for ($i = 1; $i -le $Count; $i++) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $fileName = "screenshot_{0:D3}_{1}.png" -f $i, $timestamp
    $filePath = Join-Path $resolvedOutputDir $fileName

    $bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)

    try {
        $graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
        $bitmap.Save($filePath, [System.Drawing.Imaging.ImageFormat]::Png)
        Write-Host "[$i/$Count] Saved $filePath"
    }
    finally {
        $graphics.Dispose()
        $bitmap.Dispose()
    }

    if ($i -lt $Count) {
        Start-Sleep -Seconds $IntervalSeconds
    }
}

Write-Host "Done."
