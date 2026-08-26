import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { Button, Card, EmptyState, Input, Metric, Select, Skeleton, Toast, Toolbar } from '../src/components/ui'

function scanA11y(container: HTMLElement) {
  const violations: string[] = []
  container.querySelectorAll('button').forEach((button) => {
    if (!button.textContent?.trim() && !button.getAttribute('aria-label')) violations.push('button-name')
  })
  container.querySelectorAll('input, select, textarea').forEach((field) => {
    const id = field.getAttribute('id')
    const hasLabel = field.getAttribute('aria-label') || (id && container.querySelector(`label[for="${id}"]`))
    if (!hasLabel) violations.push('form-field-label')
  })
  container.querySelectorAll('img').forEach((img) => {
    if (!img.hasAttribute('alt')) violations.push('image-alt')
  })
  return violations
}

describe('design system primitives', () => {
  it('renders the documented primitive set without accessibility violations', () => {
    const { container } = render(
      <Card aria-label="Primitive sample">
        <Toolbar aria-label="Actions"><Button>Save</Button></Toolbar>
        <label htmlFor="name">Name</label><Input id="name" />
        <Select aria-label="Region"><option>Local mesh</option></Select>
        <Metric label="Throughput" value={12.4} unit="tok/s" tier="measured" />
        <Skeleton className="h-6" />
        <EmptyState title="Nothing here" body="The API has not returned records for this surface." />
        <Toast>Saved</Toast>
      </Card>,
    )
    expect(scanA11y(container)).toEqual([])
  })

  it('proves the accessibility budget catches a real violation', () => {
    const { container } = render(<button type="button" aria-label="" />)
    expect(scanA11y(container)).toContain('button-name')
  })

  it('requires Metric tier labels at compile-time and runtime', () => {
    render(<Metric label="Spend" value={4.2} unit="mesh-seconds" tier="measured" />)
    expect(screen.getByText('measured')).toBeInTheDocument()
    // @ts-expect-error Metric numbers must carry a tier label.
    const invalid = <Metric label="Bad" value={1} />
    expect(invalid).toBeTruthy()
    expect(() => render(<Metric label="Bad" value={1} tier={undefined as never} />)).toThrow(/tier/i)
  })
})
