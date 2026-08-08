import React from 'react';
import { diff_match_patch as DiffMatchPatch } from 'diff-match-patch';

const dmp = new DiffMatchPatch();

export function renderDiffText(inputText, outputText) {
  const diffs = dmp.diff_main(inputText, outputText);
  dmp.diff_cleanupSemantic(diffs);

  return diffs.map(([operation, text], index) => {
    if (operation === 1) {
      return (
        <mark key={index} className="diff-added">
          {text}
        </mark>
      );
    }
    if (operation === -1) {
      return (
        <del key={index} className="diff-removed" data-text={text} />
      );
    }
    // 0 = Unchanged
    return <span key={index}>{text}</span>;
  });
}