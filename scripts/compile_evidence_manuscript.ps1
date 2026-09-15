# Native Word layout/export of the repository copy. Source manuscript is untouched.
$ErrorActionPreference = 'Stop'
$paperRoot = Split-Path -Parent $PSScriptRoot
$paperPath = Join-Path $paperRoot 'docs/manuscript/iom3d_hbm_evidence.docx'
$paperPDF = Join-Path $paperRoot 'docs/manuscript/iom3d_hbm_evidence.pdf'
$paperSVG = Join-Path $paperRoot 'runs/physical_decode_execution_anatomy/figure.svg'
$paperWord = New-Object -ComObject Word.Application
$paperWord.Visible = $false
$paperWord.DisplayAlerts = 0
try {
    $paperDoc = $paperWord.Documents.Open($paperPath)
    $figureRange = $paperDoc.Content.Duplicate
    if ($figureRange.Find.Execute('PHYSICAL_EXECUTION_FIGURE_PLACEHOLDER')) {
        $breakRange = $paperDoc.Range($figureRange.Start, $figureRange.Start)
        $breakRange.InsertBreak(3) # Continuous section, full-width evidence figure.
        $conclusionRange = $paperDoc.Content.Duplicate
        [void]$conclusionRange.Find.Execute('Conclusion')
        $breakRange = $paperDoc.Range($conclusionRange.Start, $conclusionRange.Start)
        $breakRange.InsertBreak(3)
        $figureRange = $paperDoc.Content.Duplicate
        [void]$figureRange.Find.Execute('PHYSICAL_EXECUTION_FIGURE_PLACEHOLDER')
        $figureSection = $figureRange.Sections.Item(1)
        $figureSection.PageSetup.TextColumns.SetCount(1)
        $conclusionRange = $paperDoc.Content.Duplicate
        [void]$conclusionRange.Find.Execute('Conclusion')
        $conclusionRange.Sections.Item(1).PageSetup.TextColumns.SetCount(2)
        $figureRange.Text = ''
        $picture = $paperDoc.InlineShapes.AddPicture($paperSVG, $false, $true, $figureRange)
        $picture.LockAspectRatio = -1
        $picture.Width = 510
        $picture.Range.ParagraphFormat.Alignment = 1
        $picture.Range.ParagraphFormat.FirstLineIndent = 0
    }
    $referenceRange = $paperDoc.Content.Duplicate
    if ($referenceRange.Find.Execute('REFERENCES')) {
        $referenceRange.ListFormat.RemoveNumbers()
    }
    foreach ($inlineFigure in $paperDoc.InlineShapes) {
        $inlineFigure.Range.ParagraphFormat.KeepWithNext = -1
    }
    $paperDoc.Repaginate()
    $paperDoc.Save()
    $paperDoc.ExportAsFixedFormat($paperPDF, 17)
    Write-Output ('PDF pages: ' + $paperDoc.ComputeStatistics(2))
    Write-Output ('PDF: ' + $paperPDF)
    $paperDoc.Close(0)
} finally {
    $paperWord.Quit()
    [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($paperWord)
}
