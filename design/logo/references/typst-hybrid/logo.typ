// LLM-Rosetta logo — Typst (real silhouette embedded + native inscription overlay)
#set page(width: 300pt, height: 300pt, margin: 0pt, fill: none)

#let ink = rgb("#cdd8e6")
#let token = rgb("#38bdf8")
#let divider = rgb("#9aa6b8")
#let lh = 7pt
#let rad = 3pt

// canvas is 300x300; the stone svg is square, drawn full-bleed
#let line(x, y, w, fill: ink, op: 85%) = place(top + left, dx: x, dy: y,
  rect(width: w, height: lh, radius: rad, fill: fill.transparentize(100% - op)))

#let divln(x, y, w) = place(top + left, dx: x, dy: y,
  rect(width: w, height: 1.6pt, radius: 1pt, fill: divider.transparentize(60%)))

#box(width: 300pt, height: 300pt)[
  // 1. real Rosetta silhouette
  #place(top + left, image("stone.svg", width: 300pt, height: 300pt))

  // 2. inscription overlay (coords tuned to the real interior ~ x56..250, y70..210)
  // TOP hieroglyphic
  #{
    let rows = ((11pt,8pt,14pt,9pt,12pt,7pt,13pt),(9pt,13pt,8pt,12pt,10pt,14pt,9pt),(13pt,9pt,11pt,13pt,8pt,12pt,10pt))
    let y = 72pt
    for row in rows {
      let x = 80pt
      for w in row { line(x, y, w); x += w + 6pt }
      y += 11pt
    }
  }
  #divln(70pt, 112pt, 168pt)
  // MIDDLE demotic + token slab
  #{
    let midw = (150pt,146pt,152pt,142pt)
    let y = 124pt
    for (i,w) in midw.enumerate() {
      if i == 2 {
        line(64pt, y, 36pt)
        place(top+left, dx: 106pt, dy: y - 1.5pt, rect(width: 46pt, height: lh+3pt, radius: rad, fill: token))
        line(158pt, y, 50pt)
      } else { line(64pt, y, w) }
      y += 9.5pt
    }
  }
  #divln(68pt, 168pt, 172pt)
  // BOTTOM greek
  #{
    let botw = (152pt,148pt,150pt,120pt)
    let y = 180pt
    for w in botw { line(62pt, y, w); y += 11pt }
  }
]
