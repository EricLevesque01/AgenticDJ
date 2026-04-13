/**
 * ReasoningCard — Graph RAG Explainability Display
 * ==================================================
 * Displays the Curator's structured reasoning for WHY the current track
 * was selected. Expandable card below the TriviaCard.
 *
 * Graph RAG explainability (Diamantini et al., 2026):
 *   "Providing clear, user-centric explanations is critical for improving
 *    transparency, facilitating informed decision-making."
 *
 * References:
 *   - Paper §4.4 (Explainable ranking)
 *   - Spec §13 (Frontend Components)
 */

'use client';

import { useEffect, useState } from 'react';
import type { CuratorReasoning } from '@/lib/types';

interface ReasoningCardProps {
  reasoning: CuratorReasoning | null;
}

export default function ReasoningCard({ reasoning }: ReasoningCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [visible, setVisible] = useState(false);
  const [displayedReasoning, setDisplayedReasoning] = useState<CuratorReasoning | null>(null);

  useEffect(() => {
    if (!reasoning) {
      setVisible(false);
      setTimeout(() => setDisplayedReasoning(null), 400);
      return;
    }

    setVisible(false);
    setTimeout(() => {
      setDisplayedReasoning(reasoning);
      setVisible(true);
    }, 100);
  }, [reasoning]);

  if (!displayedReasoning) return null;

  return (
    <div
      className="reasoning-card"
      id="reasoning-card"
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? 'translateY(0)' : 'translateY(8px)',
        transition: 'opacity 300ms ease, transform 300ms ease',
        background: 'rgba(255, 255, 255, 0.04)',
        borderRadius: '12px',
        padding: '12px 16px',
        marginTop: '8px',
        border: '1px solid rgba(255, 255, 255, 0.06)',
        cursor: 'pointer',
      }}
      onClick={() => setExpanded(!expanded)}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          fontSize: '0.7rem',
          fontWeight: 600,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          color: 'var(--color-accent-secondary, #a78bfa)',
        }}
      >
        <span aria-hidden="true">🧠</span>
        <span>Graph RAG Reasoning</span>
        <span
          style={{
            marginLeft: 'auto',
            fontSize: '0.65rem',
            opacity: 0.5,
            transition: 'transform 200ms ease',
            transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
          }}
        >
          ▼
        </span>
      </div>

      {/* Summary (always visible) */}
      <div
        style={{
          marginTop: '6px',
          fontSize: '0.85rem',
          color: 'rgba(255, 255, 255, 0.8)',
          lineHeight: 1.4,
        }}
      >
        {displayedReasoning.reasoning}
      </div>

      {/* Expanded: Ranking details */}
      {expanded && displayedReasoning.ranking && displayedReasoning.ranking.length > 0 && (
        <div
          style={{
            marginTop: '10px',
            borderTop: '1px solid rgba(255, 255, 255, 0.06)',
            paddingTop: '8px',
          }}
        >
          <div
            style={{
              fontSize: '0.7rem',
              fontWeight: 600,
              color: 'rgba(255, 255, 255, 0.5)',
              marginBottom: '6px',
            }}
          >
            CANDIDATE RANKING
          </div>
          {displayedReasoning.ranking.map((item, i) => (
            <div
              key={i}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '4px 0',
                fontSize: '0.8rem',
                color: i === 0 ? 'var(--color-accent-primary, #34d399)' : 'rgba(255, 255, 255, 0.6)',
              }}
            >
              <span
                style={{
                  fontWeight: 700,
                  minWidth: '20px',
                  fontSize: '0.75rem',
                  opacity: 0.6,
                }}
              >
                #{i + 1}
              </span>
              <span style={{ fontWeight: i === 0 ? 600 : 400 }}>
                {item.track} — {item.artist}
              </span>
              <span
                style={{
                  marginLeft: 'auto',
                  fontSize: '0.7rem',
                  opacity: 0.4,
                }}
              >
                {item.reason}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
