// Minimalist geometric piece set, drawn in a 60x60 cell (y grows downward,
// baseline at y=48). Every piece stands on a plain plinth. Flat angular
// bodies with one signature detail each:
//   pawn   - small slab with a rounded top
//   rook   - tower with a wider, notched battlement
//   knight - angular horse head (ear, muzzle, eye)
//   bishop - diamond on a base, with the traditional diagonal slit
//   queen  - three-spike crown with coronet dots
//   king   - tapered body under a bold cross
// "detail"/"slit" elements are recoloured per side in style.css so they stay
// visible on both white and black pieces.

const SHAPES = {
  p: `<path d="M20,48 h20 v-4 h-20 Z"/>
      <path d="M24,44 V31 A6,6 0 0 1 36,31 V44 Z"/>`,

  r: `<path d="M17,48 h26 v-4 h-26 Z"/>
      <path d="M22,44 V26 H19 V16 H24.5 V21 H27 V16 H33 V21 H35.5 V16 H41 V26 H38 V44 Z"/>`,

  n: `<path d="M16,48 h28 v-4 h-28 Z"/>
      <path d="M20,44 L38,44 L38,30 L36,18 L34,10 L28,10 L24,16 L12,26 L12,31 L22,33 Z"/>
      <circle class="detail" cx="27.2" cy="19" r="1.8"/>`,

  b: `<path d="M18,48 h24 v-4 h-24 Z"/>
      <path d="M30,12 L40,28 L30,44 L20,28 Z"/>
      <path class="slit" d="M27,20 L34,27"/>`,

  q: `<path d="M15,48 h30 v-4 h-30 Z"/>
      <path d="M23,44 L37,44 L34,30 L26,30 Z"/>
      <path d="M21,30 L39,30 L37,13 L32,22 L30,11 L28,22 L23,13 Z"/>
      <circle cx="23" cy="9" r="2.1"/>
      <circle cx="30" cy="6.5" r="2.1"/>
      <circle cx="37" cy="9" r="2.1"/>`,

  k: `<path d="M15,48 h30 v-4 h-30 Z"/>
      <path d="M22,44 L38,44 L34,26 L26,26 Z"/>
      <path d="M27.5,24 V17 H24 V12 H27.5 V6 H32.5 V12 H36 V17 H32.5 V24 Z"/>`,
};

export function pieceMarkup(type) {
  return SHAPES[type] ?? '';
}
