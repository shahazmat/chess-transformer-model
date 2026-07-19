// SVG chessboard: rendering + click reporting. Stateless — app.js hands it a
// snapshot to draw and receives square names ('e4') back on click.

import { pieceMarkup } from './pieces.js';

const SQ = 60;
const SIZE = 8 * SQ;
const FILES = [...'abcdefgh'];

export function createBoard(svg, onSquareClick) {
  svg.setAttribute('viewBox', `0 0 ${SIZE} ${SIZE}`);
  let current = null;

  svg.addEventListener('click', (e) => {
    if (!current) return;
    const rect = svg.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * SIZE;
    const y = ((e.clientY - rect.top) / rect.height) * SIZE;
    const col = Math.min(7, Math.max(0, Math.floor(x / SQ)));
    const row = Math.min(7, Math.max(0, Math.floor(y / SQ)));
    const file = current.orientation === 'w' ? col : 7 - col;
    const rank = current.orientation === 'w' ? 7 - row : row;
    onSquareClick(FILES[file] + (rank + 1));
  });

  // Top-left corner of a square in board pixels, respecting orientation.
  function xy(square, orientation) {
    const file = square.charCodeAt(0) - 97;
    const rank = square.charCodeAt(1) - 49;
    const x = (orientation === 'w' ? file : 7 - file) * SQ;
    const y = (orientation === 'w' ? 7 - rank : rank) * SQ;
    return [x, y];
  }

  // state = { board, orientation, selected, targets, lastMove, checkSquare }
  //   board:       Chess#board() 8x8 array (rank 8 first)
  //   targets:     Map<square, 'move' | 'capture'>
  //   lastMove:    { from, to } | null
  //   checkSquare: square of the king in check | null
  function render(state) {
    current = state;
    const { orientation } = state;
    const parts = [];

    for (let file = 0; file < 8; file++) {
      for (let rank = 0; rank < 8; rank++) {
        const square = FILES[file] + (rank + 1);
        const [x, y] = xy(square, orientation);
        const shade = (file + rank) % 2 === 0 ? 'dark' : 'light';
        parts.push(`<rect class="sq ${shade}" x="${x}" y="${y}" width="${SQ}" height="${SQ}"/>`);
      }
    }

    // Coordinate labels along the user's bottom and left edges.
    const bottomRank = orientation === 'w' ? 1 : 8;
    const leftFile = orientation === 'w' ? 'a' : 'h';
    for (let file = 0; file < 8; file++) {
      const square = FILES[file] + bottomRank;
      const [x, y] = xy(square, orientation);
      const shade = (file + bottomRank - 1) % 2 === 0 ? 'on-dark' : 'on-light';
      parts.push(`<text class="coord ${shade}" x="${x + SQ - 5}" y="${y + SQ - 5}" text-anchor="end">${FILES[file]}</text>`);
    }
    for (let rank = 1; rank <= 8; rank++) {
      const square = leftFile + rank;
      const [x, y] = xy(square, orientation);
      const shade = (leftFile.charCodeAt(0) - 97 + rank - 1) % 2 === 0 ? 'on-dark' : 'on-light';
      parts.push(`<text class="coord ${shade}" x="${x + 5}" y="${y + 15}">${rank}</text>`);
    }

    if (state.lastMove) {
      for (const square of [state.lastMove.from, state.lastMove.to]) {
        const [x, y] = xy(square, orientation);
        parts.push(`<rect class="last-move" x="${x}" y="${y}" width="${SQ}" height="${SQ}"/>`);
      }
    }
    if (state.checkSquare) {
      const [x, y] = xy(state.checkSquare, orientation);
      parts.push(`<rect class="check" x="${x}" y="${y}" width="${SQ}" height="${SQ}"/>`);
    }
    if (state.selected) {
      const [x, y] = xy(state.selected, orientation);
      parts.push(`<rect class="selected" x="${x + 2}" y="${y + 2}" width="${SQ - 4}" height="${SQ - 4}"/>`);
    }

    for (const row of state.board) {
      for (const cell of row) {
        if (!cell) continue;
        const [x, y] = xy(cell.square, orientation);
        const color = cell.color === 'w' ? 'white' : 'black';
        parts.push(`<g class="piece ${color}" transform="translate(${x},${y})">${pieceMarkup(cell.type)}</g>`);
      }
    }

    if (state.targets) {
      for (const [square, kind] of state.targets) {
        const [x, y] = xy(square, orientation);
        parts.push(kind === 'capture'
          ? `<circle class="ring" cx="${x + SQ / 2}" cy="${y + SQ / 2}" r="${SQ / 2 - 5}"/>`
          : `<circle class="dot" cx="${x + SQ / 2}" cy="${y + SQ / 2}" r="9"/>`);
      }
    }

    svg.innerHTML = parts.join('');
  }

  return { render };
}
