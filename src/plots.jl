using Plots, StatsPlots, DataFrames, LaTeXStrings

default_lbls = Dict("T" => L"$T \quad(cells/mm^3)$","T_inf" => L"$T_{inf} \quad(cells/mm^3)$", "M" => L"$M \quad(cells/mm^3)$", "M_inf" => L"$M_{inf}\quad(cells/mm^3)$", "V" => L"V \quad (copies/ml)")
default_yrange = Dict("T" => (1.0,1.0e3),"T_inf" => (0.1,1.1e2), "M" => (0.0,1.5e4), "M_inf" => (0.0,1.0e4), "V" => (0.0,2.5e3))

function HIV_plot(Xarray::AbstractArray; size=(600, 600), linecolor=:steelblue, lbl=[L"Noisy \quad measurements" L"RHONN"], lbls=default_lbls, title=L"HIV \quad Dynamics \quad Identification", linealpha=[1.0 0.8 0.4], dt=2, ls=[:solid :dash :dashdot], yrange=default_yrange)
    length(Xarray) == length(lbl) ? nothing : lbl = false
    time_span = 0:dt:(2*nrow(first(Xarray)) - 1)
    
    l = @layout [a ; b c]
    p1 = plot(time_span, [x[:,"T"] for x in Xarray], yscale=:log10, yrange=yrange["T"], ylabel=lbls["T"], lw=[2 for i in 1:length(Xarray)], ls=ls, label=lbl, legend=false, linealpha=linealpha, linecolor=linecolor)
    p2 = plot(time_span, [x[:,"T_inf"] for x in Xarray], yscale=:log10, yrange=yrange["T_inf"], ylabel=lbls["T_inf"], lw=[2 for i in 1:length(Xarray)], ls=ls, label=lbl, xlabel=L"Time \quad(days)", legend=false, linealpha=linealpha, linecolor=linecolor)
    p3 = plot(p1, p2, layout=(2,1))
    p1 = plot(time_span, [x[:,"M"] for x in Xarray], yrange=yrange["M"], ylabel=lbls["M"], lw=[2 for i in 1:length(Xarray)], ls=ls, label=lbl, legend=false, linealpha=linealpha, linecolor=linecolor)
    p2 = plot(time_span, [x[:,"M_inf"] for x in Xarray], yrange=yrange["M_inf"], ylabel=lbls["M_inf"], lw=[2 for i in 1:length(Xarray)], ls=ls, label=lbl, xlabel=L"Time \quad(days)", legend=false, linealpha=linealpha, linecolor=linecolor)
    p4 = plot(p1, p2, layout=(2,1))
    p5 = plot(time_span, [x[:,"V"] for x in Xarray], yrange=yrange["V"], ylabel=lbls["V"], lw=[2 for i in 1:length(Xarray)], ls=ls, label=lbl, xlabel=L"Time \quad(days)", title=title, linealpha=linealpha, linecolor=linecolor)
    #savefig(p, joinpath(plotsdir(), "HIV_virus.png"))
    plot(p5, p3, p4, layout = l, size = size)
end
HIV_plot(X::DataFrame; size=(600, 600), linecolor=:steelblue, yrange=default_yrange, lbl=["HIV model" "RHONN"], lbls=default_lbls, title=false, ls=[:solid :dash :dashdot]) = HIV_plot([X]; size=size, lbl=lbl, title=title, linecolor=linecolor, ls=ls, lbls=lbls, yrange=yrange)
