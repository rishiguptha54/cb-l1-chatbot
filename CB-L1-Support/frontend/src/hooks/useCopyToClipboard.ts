import { useCallback, useState } from "react";

/** Copy text to the clipboard and expose a transient "copied" flag. */
export function useCopyToClipboard(resetAfter = 1800): [boolean, (text: string) => void] {
  const [copied, setCopied] = useState(false);

  const copy = useCallback(
    (text: string) => {
      const done = () => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), resetAfter);
      };
      if (navigator.clipboard?.writeText) {
        navigator.clipboard.writeText(text).then(done).catch(() => fallbackCopy(text, done));
      } else {
        fallbackCopy(text, done);
      }
    },
    [resetAfter],
  );

  return [copied, copy];
}

function fallbackCopy(text: string, done: () => void) {
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
    done();
  } catch {
    /* ignore */
  }
}
