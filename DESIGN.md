# MedTutor Design System

## Direction

Research Workspace: restrained dark scientific product UI inspired by Perplexity, Linear, and scientific paper readers.

## Tokens

```css
--background: oklch(0.16 0.018 245);
--surface-1: oklch(0.19 0.018 245);
--surface-2: oklch(0.22 0.019 245);
--border: oklch(0.31 0.020 245);
--text-primary: oklch(0.93 0.010 230);
--text-secondary: oklch(0.70 0.018 235);
--accent: oklch(0.76 0.125 205);
--success: oklch(0.72 0.115 155);
--warning: oklch(0.80 0.115 85);
--danger: oklch(0.68 0.155 25);
```

## Typography

Inter or system sans. Fixed product scale, body 14-16px, page heading 20-24px, line length 68-72ch, letter spacing 0.

## Geometry

Radius 4px, 6px, or 8px. Borders 1px. Controls 32px compact or 40px standard. Lucide icons at 16px or 18px.

## Layout

Desktop uses a persistent left navigation, flexible main workspace, and optional right evidence inspector. Tablet and mobile convert secondary panels to drawers.

## Interaction

Transitions last 150-220ms and communicate state. Use opacity and transform only. Respect reduced motion.

## Accessibility

WCAG AA contrast, complete keyboard navigation, visible focus rings, semantic headings, descriptive citation labels, focus-managed drawers, and live status announcements.

## Bans

No gradients, glassmorphism, oversized hero text, nested cards, colored side stripes, decorative blobs, or raw JSON in user-facing views.

