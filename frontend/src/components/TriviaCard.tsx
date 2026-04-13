/**
 * TriviaCard — Knowledge Graph Link Display
 * ==========================================
 * Displays the trivia link between artists when received from server.
 * Fades in with 300ms animation, persists until next track change.
 *
 * References:
 *   - Spec §13.2 (Trivia Link Card)
 *   - Spec §7.3 ({ type: "trivia", data: { link_type, description } })
 */

'use client';

import { useEffect, useRef, useState } from 'react';

export interface TriviaData {
  link_type: string;
  description: string;
}

interface TriviaCardProps {
  trivia: TriviaData | null;
}

const LINK_TYPE_ICONS: Record<string, string> = {
  shared_producer: '🎛️',
  same_studio:     '🏛️',
  genre_movement:  '🎵',
  influence:       '⚡',
};

const LINK_TYPE_LABELS: Record<string, string> = {
  shared_producer: 'Shared Producer',
  same_studio:     'Same Studio',
  genre_movement:  'Genre Connection',
  influence:       'Musical Influence',
};

export default function TriviaCard({ trivia }: TriviaCardProps) {
  const [visible, setVisible] = useState(false);
  const [displayedTrivia, setDisplayedTrivia] = useState<TriviaData | null>(null);
  const prevTriviaRef = useRef<TriviaData | null>(null);

  useEffect(() => {
    if (!trivia) {
      // Fade out
      setVisible(false);
      setTimeout(() => setDisplayedTrivia(null), 400);
      prevTriviaRef.current = null;
      return;
    }

    // Only animate if it's a genuinely new trivia item
    if (trivia.description !== prevTriviaRef.current?.description) {
      setVisible(false);
      setTimeout(() => {
        setDisplayedTrivia(trivia);
        setVisible(true);
      }, 100);
      prevTriviaRef.current = trivia;
    }
  }, [trivia]);

  if (!displayedTrivia) return <div className="trivia-card" id="trivia-card" />;

  const icon = LINK_TYPE_ICONS[displayedTrivia.link_type] ?? '🔗';
  const label = LINK_TYPE_LABELS[displayedTrivia.link_type] ?? 'Musical Link';

  return (
    <div className="trivia-card" id="trivia-card">
      <div
        className="trivia-card__content"
        style={{
          opacity: visible ? 1 : 0,
          transform: visible ? 'translateY(0)' : 'translateY(8px)',
          transition: 'opacity 300ms ease, transform 300ms ease',
        }}
      >
        <span className="trivia-card__icon" aria-hidden="true">{icon}</span>
        <div>
          <div
            style={{
              fontSize: '0.7rem',
              fontWeight: 600,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: 'var(--color-accent-primary)',
              marginBottom: 4,
            }}
          >
            {label}
          </div>
          <div>{displayedTrivia.description}</div>
        </div>
      </div>
    </div>
  );
}
