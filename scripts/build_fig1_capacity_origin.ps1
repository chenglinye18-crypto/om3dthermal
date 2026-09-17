$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$out=Join-Path $root 'runs\background_fig1_origin'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$source=Import-Csv (Join-Path $root 'runs\formal_long_context_v3\capacity_audit.csv')
$rows=@($source | Where-Object {$_.H -eq '126000'} | ForEach-Object {
 $w=[double]$_.weight_bytes/1e9;$k=[double]$_.final_KV_bytes/1e9
 [pscustomobject]@{Model=$_.model;Batch=[int]$_.B;Model_Weights_GB=$w;KV_Cache_GB=$k;Total_GB=$w+$k;H200_HBM_Capacity_GB=[double]$_.HBM_capacity_bytes/1e9}
})
$rows | Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $out 'capacity_data.csv')
$o=New-Object -ComObject Origin.Application
function LT([string]$s){if(-not $o.Execute($s)){throw "Origin command failed: $s"}}
Write-Output ('Origin version: '+$o.Evaluate('@V'))
LT 'newbook name:=Capacity_Data sheet:=1; wks.ncols=8;'
LT 'wks.col1.lname$="Model"; wks.col2.lname$="Batch"; wks.col3.lname$="Model Weights"; wks.col4.lname$="KV Cache"; wks.col5.lname$="Total_GB"; wks.col6.lname$="H200_HBM_Capacity_GB"; wks.col7.lname$="Plot position"; wks.col8.lname$="Batch label";'
$positions=@(1,2,3,4.5,5.5,6.5)
for($i=0;$i -lt 6;$i++){$r=$rows[$i];$n=$i+1;LT ('col(A)['+$n+']$="'+$r.Model+'"; col(B)['+$n+']='+$r.Batch+'; col(C)['+$n+']='+$r.Model_Weights_GB.ToString('R',[cultureinfo]::InvariantCulture)+'; col(D)['+$n+']='+$r.KV_Cache_GB.ToString('R',[cultureinfo]::InvariantCulture)+'; col(E)['+$n+']='+$r.Total_GB.ToString('R',[cultureinfo]::InvariantCulture)+'; col(F)['+$n+']='+$r.H200_HBM_Capacity_GB.ToString('R',[cultureinfo]::InvariantCulture)+'; col(G)['+$n+']='+$positions[$i]+'; col(H)['+$n+']$="B'+$r.Batch+'";')}
LT 'plotxy iy:=(7,3:4) plot:=213 ogl:=[<new template:=stackcolumn name:=CapacityChart>];'
LT 'layer.x.type=1;layer.y.type=1;layer.x.thickness=1;layer.y.thickness=1;layer.x.tickthickness=1;layer.y.tickthickness=1;layer.x.minorticks=0;layer.y.minorticks=0;layer.x.ticks=1;layer.y.ticks=1;layer.x.from=0.3;layer.x.to=7.2;layer.y.from=0;layer.y.to=1280;layer.y.inc=250;'
LT 'layer.x.ticksbydata$="1 2 3 4.5 5.5 6.5";'
LT 'layer.x.label.type=8;layer.x.label.dataset$="Book1_H";layer.x.label.font=font(Times New Roman);layer.y.label.font=font(Times New Roman);'
LT 'label -yl "Working-set Capacity (GB)"; label -xb "";'
LT 'legend.text$="\l(1) Model Weights     \l(2) KV Cache";legend.font=font(Times New Roman);legend.fsize=18;legend.background=0;'
LT 'set Book1_C -gm 1;set Book1_C -pfb color(128,104,140);set Book1_D -pfb color(209,154,85);set Book1_C -pbc color(50,50,50);set Book1_D -pbc color(50,50,50);set Book1_C -vw 0.7;set Book1_D -vw 0.7;set Book1_C -vg 30;set Book1_D -vg 30;'
LT 'page.width=9000;page.height=6000;layer.left=17;layer.top=16;layer.width=78;layer.height=65;'
LT 'layer.x.label.pt=18;layer.y.label.pt=18;yl.font=font(Times New Roman);yl.fsize=20;'
LT 'legend.x=3.75;legend.y=1400;legend.attach=2;legend.x=3.75;legend.y=1400;'
LT 'label -n Model1 "Llama-3.1-8B"; Model1.font=font(Times New Roman);Model1.fsize=18;Model1.x=2;Model1.y=-175;'
LT 'label -n Model2 "Qwen2.5-32B"; Model2.font=font(Times New Roman);Model2.fsize=18;Model2.x=5.5;Model2.y=-175;'
for($i=0;$i -lt 6;$i++){$n=$i+1;$r=$rows[$i];LT ('label -n Total'+$n+' "'+$r.Total_GB.ToString('F1',[cultureinfo]::InvariantCulture)+'"; Total'+$n+'.font=font(Times New Roman);Total'+$n+'.fsize=18;Total'+$n+'.x='+$positions[$i]+';Total'+$n+'.y='+($r.Total_GB+35).ToString('R',[cultureinfo]::InvariantCulture)+';')}
$cap=$rows[0].H200_HBM_Capacity_GB.ToString('R',[cultureinfo]::InvariantCulture)
LT ('draw -n HBMCapacity -w 1.4 -d 1 -l -h '+$cap+';HBMCapacity.color=color(145,72,65);')
LT 'label -n HBMLabel "H200 HBM Capacity";HBMLabel.font=font(Times New Roman);HBMLabel.fsize=16;HBMLabel.color=color(145,72,65);HBMLabel.x=3.95;HBMLabel.y=210;'
$path=$out.Replace('\','/')
LT ('expGraph type:=pdf filename:="fig1_capacity_origin" path:="'+$path+'" overwrite:=replace;')
LT ('expGraph type:=png filename:="fig1_capacity_origin" path:="'+$path+'" overwrite:=replace;')
if(-not $o.Save((Join-Path $out 'fig1_capacity_origin.opju'))){throw 'Save failed'}
$o.Exit() | Out-Null







# Origin 2021 predates native SVG export. Convert its vector PDF without redrawing.
& 'C:\Users\Leslie\Miniconda3\envs\om3dthermal\python.exe' -c "import pathlib,sys,pymupdf; p=pathlib.Path(sys.argv[1]); d=pymupdf.open(p.with_suffix('.pdf')); p.with_suffix('.svg').write_text(d[0].get_svg_image(text_as_path=False),encoding='utf-8')" (Join-Path $out 'fig1_capacity_origin')
if ($LASTEXITCODE -ne 0) { throw 'PDF-to-SVG conversion failed' }
