import { useMemo, useState } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { fmt, usd, short } from './ui';

const severity = (item) => {
  if (item.status === 'Critical') return 0;
  if (item.status === 'Reorder') return 1;
  if (item.stockout_risk === 'High' || item.stockout_risk === 'Elevated') return 2;
  if (item.status === 'Overstock') return 3;
  return 4;
};

const labelFor = (item) => {
  if (item.status === 'Critical') return 'Running Low';
  if (item.status === 'Reorder') return 'Reorder Now';
  if (item.status === 'Overstock') return 'Overstocked';
  if (item.stockout_risk === 'High' || item.stockout_risk === 'Elevated') return 'Running Low';
  return 'Healthy';
};

const messageFor = (item) => {
  const label = labelFor(item);
  if (label === 'Running Low') return 'Stock may run out before the next delivery.';
  if (label === 'Reorder Now') return 'Stock is below the level needed for the next delivery.';
  if (label === 'Overstocked') return 'There is more stock than expected demand needs.';
  return 'Enough stock for now.';
};

const statusClass = (item) => item.status || 'Healthy';
const daysText = (value) => value == null || !Number.isFinite(Number(value)) ? '—' : `${fmt(value)} days`;

function SalesOverview({ data }) {
  const chartData = data || [];
  return <section className="owner-panel owner-sales" aria-labelledby="owner-sales-title">
    <div className="owner-panel-heading">
      <div><h2 id="owner-sales-title">Sales Overview</h2><p>Recent sales and the forecast</p></div>
      <div className="owner-legend"><span><i className="actual" />Actual sales</span><span><i className="predicted" />Predicted sales</span></div>
    </div>
    {chartData.length ? <div className="owner-chart">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 8, right: 12, left: -18, bottom: 0 }}>
          <CartesianGrid stroke="#e8ece5" vertical={false} />
          <XAxis dataKey="date" tickFormatter={short} tick={{ fontSize: 11, fill: '#738078' }} axisLine={false} tickLine={false} minTickGap={24} />
          <YAxis tick={{ fontSize: 11, fill: '#738078' }} axisLine={false} tickLine={false} width={42} />
          <Tooltip labelFormatter={short} formatter={(value, name) => [fmt(value), name === 'actual' ? 'Actual sales' : 'Predicted sales']} contentStyle={{ borderRadius: 8, borderColor: '#dfe5dc', fontSize: 12 }} />
          <Line type="monotone" dataKey="actual" name="actual" stroke="#397b53" strokeWidth={2} dot={false} connectNulls />
          <Line type="monotone" dataKey="pred" name="pred" stroke="#93a879" strokeWidth={2} strokeDasharray="5 4" dot={false} connectNulls />
        </LineChart>
      </ResponsiveContainer>
    </div> : <div className="owner-empty">Sales history is not available yet.</div>}
  </section>;
}

export default function Owner({ d }) {
  const [sel, setSel] = useState(d?.items?.[0]?.id ?? null);
  const [overrides, setOverrides] = useState({});
  const [sort, setSort] = useState({ key: 'priority', dir: 1 });

  const items = d?.items || [];
  const k = d?.kpis || {};
  const selected = items.find((item) => item.id === sel) || items[0];

  const attention = useMemo(() => [...items]
    .filter((item) => severity(item) < 4)
    .sort((a, b) => severity(a) - severity(b) || (Number(a.cover_days) || 0) - (Number(b.cover_days) || 0))
    .slice(0, 5), [items]);

  const sortedItems = useMemo(() => [...items].sort((a, b) => {
    const { key, dir } = sort;
    if (key === 'priority') return (severity(a) - severity(b)) * dir;
    if (key === 'name') return String(a.name || a.id).localeCompare(String(b.name || b.id)) * dir;
    if (key === 'status') return (severity(a) - severity(b)) * dir;
    const av = Number(a[key]); const bv = Number(b[key]);
    return ((Number.isFinite(av) ? av : -Infinity) - (Number.isFinite(bv) ? bv : -Infinity)) * dir;
  }), [items, sort]);

  const detail = useMemo(() => {
    if (!selected) return null;
    const override = overrides[selected.id] || {};
    const stock = Math.max(0, Number(override.stock ?? selected.stock) || 0);
    const onOrder = Math.max(0, Number(override.on_order ?? selected.on_order) || 0);
    const lead = Math.max(1, Number(override.lead ?? selected.lead) || 1);
    const service = Number(override.service_level ?? selected.service_level ?? d.service_level) || d.service_level;
    const unitCost = Math.max(0, Number(override.cost ?? selected.cost) || 0);
    const forecastRows = (selected.series || []).filter((row) => row.pred !== undefined);
    const z = service >= 0.99 ? 2.326 : service >= 0.95 ? 1.645 : 1.282;
    const safety = Math.ceil(z * Math.sqrt((selected.safety ** 2) / (1.645 ** 2) * (lead / selected.lead)));
    const leadDemand = forecastRows.slice(0, lead).reduce((sum, row) => sum + row.pred, 0);
    const upperDemand = forecastRows.slice(0, lead).reduce((sum, row) => sum + (row.hi ?? row.pred), 0);
    const orderQty = Math.max(0, Math.ceil((forecastRows.slice(0, lead + 7).reduce((sum, row) => sum + row.pred, 0) + safety - stock - onOrder) / 10) * 10);
    let cover = selected.cover_days;
    let remaining = stock;
    for (let index = 0; index < forecastRows.length; index += 1) {
      if (remaining < forecastRows[index].pred) { cover = index; break; }
      remaining -= forecastRows[index].pred;
    }
    const status = stock < leadDemand ? 'Critical' : stock + onOrder < leadDemand + safety ? 'Reorder' : cover > 14 ? 'Overstock' : 'Healthy';
    const risk = stock < leadDemand ? 'High' : stock < upperDemand ? 'Elevated' : 'Low';
    return { stock, onOrder, lead, service, unitCost, daily: forecastRows[0]?.pred || 0, safety, leadDemand, orderQty, cover, status, risk };
  }, [selected, overrides]);

  const updateOverride = (key, value) => {
    if (!selected) return;
    setOverrides((prev) => ({ ...prev, [selected.id]: { ...prev[selected.id], [key]: value } }));
  };

  const showDetails = (item) => {
    setSel(item.id);
    window.setTimeout(() => document.getElementById('owner-product-details')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 0);
  };

  const toggleSort = (key) => setSort((current) => ({ key, dir: current.key === key ? -current.dir : 1 }));
  const sortButton = (key, label) => <button type="button" className="owner-sort" onClick={() => toggleSort(key)} aria-label={`Sort by ${label}`} aria-sort={sort.key === key ? (sort.dir === 1 ? 'ascending' : 'descending') : 'none'}>{label}{sort.key === key ? <span aria-hidden="true">{sort.dir === 1 ? ' ↑' : ' ↓'}</span> : null}</button>;

  if (!d || !Array.isArray(d.items)) return <div className="owner-dashboard"><div className="owner-empty">Store data could not be loaded. Please refresh and try again.</div></div>;
  if (!items.length) return <div className="owner-dashboard"><div className="owner-empty">No products are available yet.</div></div>;

  const summaries = [
    { title: 'Needs Reordering', value: Number(k.critical || 0) + Number(k.reorder || 0), note: 'Products to order' },
    { title: 'Running Low', value: items.filter((item) => item.status === 'Critical' || item.stockout_risk === 'High' || item.stockout_risk === 'Elevated').length, note: 'At risk of running out' },
    { title: 'Healthy Stock', value: k.healthy ?? items.filter((item) => statusClass(item) === 'Healthy').length, note: 'Enough stock for now' },
    { title: 'Overstocked', value: k.overstock ?? items.filter((item) => item.status === 'Overstock').length, note: 'More than currently needed' },
  ];

  return <main className="owner-dashboard">
    <header className="owner-page-heading"><div><span className="owner-eyebrow">STORE OVERVIEW</span><h1>Good day. Here’s your stock at a glance.</h1><p>See what needs attention and what you may need to order.</p></div><span className="owner-product-count">{fmt(items.length)} products</span></header>

    <section className="owner-summary" aria-label="Inventory summary">
      {summaries.map((item) => <article className="owner-summary-card" key={item.title}><span>{item.title}</span><strong>{fmt(item.value)}</strong><small>{item.note}</small></article>)}
    </section>

    <div className="owner-main-grid">
      <section className="owner-panel owner-attention" aria-labelledby="owner-attention-title">
        <div className="owner-panel-heading"><div><h2 id="owner-attention-title">Needs Your Attention</h2><p>Your most urgent stock decisions</p></div><a href="#owner-all-products" className="owner-text-link">View all products <span aria-hidden="true">→</span></a></div>
        {attention.length ? <div className="owner-attention-list">{attention.map((item) => <article className="owner-attention-row" key={item.id}>
          <div className="owner-attention-product"><div className="owner-product-name">{item.name || item.id}</div><span className={`owner-status ${statusClass(item)}`}>{labelFor(item)}</span><p>{messageFor(item)}</p></div>
          <div className="owner-attention-stock"><span>Stock available</span><strong>{fmt(item.stock)}</strong></div>
          <div className="owner-attention-stock"><span>Estimated days left</span><strong>{daysText(item.cover_days)}</strong></div>
          <div className="owner-attention-order"><span>Recommended order</span><strong>{Number(item.order_qty) > 0 ? fmt(item.order_qty) : '—'}</strong></div>
          <button type="button" className="owner-view-button" onClick={() => showDetails(item)}>View details</button>
        </article>)}</div> : <div className="owner-empty">All products are in good shape. There are no urgent actions right now.</div>}
      </section>
      <SalesOverview data={d.total} />
    </div>

    <section className="owner-panel owner-products" id="owner-all-products" aria-labelledby="owner-products-title">
      <div className="owner-panel-heading"><div><h2 id="owner-products-title">All Products</h2><p>Current inventory and recommended orders</p></div><span className="owner-table-count">{fmt(items.length)} products</span></div>
      <div className="owner-table-wrap"><table className="owner-table"><thead><tr>
        <th>{sortButton('name', 'Product')}</th><th>{sortButton('stock', 'Stock Available')}</th><th>{sortButton('cover_days', 'Days Remaining')}</th><th>{sortButton('status', 'Status')}</th><th>{sortButton('order_qty', 'Recommended Order')}</th>
      </tr></thead><tbody>{sortedItems.map((item) => <tr key={item.id} className={sel === item.id ? 'selected' : ''} onClick={() => setSel(item.id)}>
        <td data-label="Product"><button type="button" className="owner-product-link" onClick={(event) => { event.stopPropagation(); setSel(item.id); }}>{item.name || item.id}</button></td>
        <td data-label="Stock Available">{fmt(item.stock)}</td><td data-label="Days Remaining">{daysText(item.cover_days)}</td>
        <td data-label="Status"><span className={`owner-status ${statusClass(item)}`}>{labelFor(item)}</span></td><td data-label="Recommended Order">{Number(item.order_qty) > 0 ? fmt(item.order_qty) : '—'}</td>
      </tr>)}</tbody></table></div>
    </section>

    {selected && detail && <section className="owner-panel owner-details" id="owner-product-details" aria-labelledby="owner-details-title">
      <div className="owner-panel-heading"><div><span className="owner-eyebrow">PRODUCT DETAILS</span><h2 id="owner-details-title">{selected.name || selected.id}</h2><p>{messageFor({ ...selected, status: detail.status, stockout_risk: detail.risk })}</p></div><span className={`owner-status ${detail.status}`}>{labelFor({ ...selected, status: detail.status, stockout_risk: detail.risk })}</span></div>
      <div className="owner-detail-metrics">
        <div><span>Stock available</span><strong>{fmt(detail.stock)}</strong></div><div><span>On order</span><strong>{fmt(detail.onOrder)}</strong></div><div><span>Estimated days remaining</span><strong>{daysText(detail.cover)}</strong></div><div><span>Recommended order</span><strong>{detail.orderQty > 0 ? fmt(detail.orderQty) : 'No order needed'}</strong></div>
      </div>
      <p className="owner-detail-note">Order estimate uses this product’s expected sales before delivery and current stock on hand and on order.</p>
      <details className="owner-assumptions"><summary>Adjust planning assumptions</summary><p>These changes update this estimate on your screen only; they do not change your saved inventory.</p>
        <div className="owner-input-grid">
          <label>Stock available<input type="number" min="0" value={detail.stock} onChange={(event) => updateOverride('stock', Number(event.target.value))} /></label>
          <label>Stock on order<input type="number" min="0" value={detail.onOrder} onChange={(event) => updateOverride('on_order', Number(event.target.value))} /></label>
          <label>Delivery lead time (days)<input type="number" min="1" value={detail.lead} onChange={(event) => updateOverride('lead', Number(event.target.value))} /></label>
          <label>Service level (%)<input type="number" min="50" max="99.9" step="0.1" value={Number(detail.service) <= 1 ? Number(detail.service) * 100 : detail.service} onChange={(event) => updateOverride('service_level', Number(event.target.value) / 100)} /></label>
          <label>Product cost<input type="number" min="0" step="0.01" value={detail.unitCost} onChange={(event) => updateOverride('cost', Number(event.target.value))} /></label>
        </div>
        <div className="owner-assumption-result"><span>Estimated purchase cost</span><strong>{usd(detail.orderQty * detail.unitCost)}</strong></div>
      </details>
    </section>}

    <details className="owner-financials"><summary>Financial overview</summary><div><span>Estimated purchase cost for recommended orders <strong>{usd(k.order_cost)}</strong></span><span>Expected sales over the next 7 days <strong>{usd(k.revenue_7)}</strong></span></div></details>
  </main>;
}
