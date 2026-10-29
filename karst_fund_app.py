"""KARST-FUND | Dashboard prototype. Run: streamlit run karst_fund_app.py"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.optimize import milp, Bounds, LinearConstraint

st.set_page_config(page_title='KARST-FUND',page_icon='💧',layout='wide')
st.title('💧 KARST-FUND')
st.caption('Dashboard simulasi pembiayaan kekeringan Gunungkidul | Prototipe penelitian, bukan estimasi anggaran resmi')
st.info('👋 **Selamat datang!** Dashboard siap digunakan dengan data historis bawaan. Atur parameter di sidebar, lalu jelajahi tab Dana cadangan, Optimisasi dana, dan Validasi target. **Tidak perlu mengunggah file.**')

POP_K={'Panggang':29.22,'Purwosari':20.87,'Paliyan':32.27,'Saptosari':38.60,'Tepus':35.31,'Tanjungsari':28.41,'Rongkop':28.78,'Girisubo':24.64,'Semanu':59.03,'Ponjong':54.93,'Karangmojo':56.25,'Wonosari':87.79,'Playen':60.91,'Patuk':34.52,'Gedangsari':38.67,'Nglipar':32.79,'Ngawen':34.23,'Semin':55.98}
REGIONS=list(POP_K)
POP=np.array(list(POP_K.values()))*1000

@st.cache_data
def read_historical(content):
    from io import BytesIO
    raw=BytesIO(content)
    monthly=pd.read_excel(raw,sheet_name='Data_Bulanan_SPI')
    raw.seek(0)
    episodes=pd.read_excel(raw,sheet_name='Episode_Kekeringan')
    return monthly,episodes

@st.cache_data(show_spinner=False)
def simulate(episode_tuple, n, seed, q, capacity, trip_cost, days_lo, days_hi, target_lo, target_mode, target_hi, delivery_lo, delivery_mode, delivery_hi, regional_sigma):
    rng=np.random.default_rng(seed)
    durations=np.array(episode_tuple,dtype=float)
    months=rng.choice(durations,size=n,replace=True)
    days=months*rng.uniform(days_lo,days_hi,n)
    base_target=rng.triangular(target_lo,target_mode,target_hi,n)
    multipliers=rng.lognormal(0,regional_sigma,size=(n,len(POP)))
    multipliers/=((multipliers*POP).sum(axis=1,keepdims=True)/POP.sum())
    target=np.clip(base_target[:,None]*multipliers,0,1)
    delivery=rng.triangular(delivery_lo,delivery_mode,delivery_hi,n)
    need=POP[None,:]*target*q*days[:,None]*delivery[:,None]
    trips=np.ceil(need/capacity)
    costs=trips*trip_cost
    return days,need,trips,costs,costs.sum(axis=1),target,delivery

def integer_fair_allocate(need, budget, capacity, trip_cost, minimum_trips=None):
    """Max-min fairness followed by maximum served liters; whole truck trips."""
    need=np.asarray(need,dtype=float)
    n=len(need)
    max_trips=np.ceil(need/capacity).astype(int)
    base=np.zeros(n,dtype=int) if minimum_trips is None else np.asarray(minimum_trips,dtype=int)
    base=np.minimum(base,max_trips)
    budget_trips=int(np.floor((budget+1e-6)/trip_cost))
    if base.sum()>budget_trips:
        raise ValueError('Anggaran tidak cukup untuk perjalanan lokal yang diwajibkan.')
    if need.sum()<=0 or budget_trips==0:
        return np.zeros(n,dtype=int),np.zeros(n)
    # Binary search the best feasible minimum coverage; monotone in lambda.
    positive=need>0
    low,high=0.0,1.0
    for _ in range(45):
        mid=(low+high)/2
        required=np.maximum(base,np.ceil(np.maximum(0,mid*need/capacity-1e-12)).astype(int))
        if required.sum()<=budget_trips:low=mid
        else:high=mid
    required=np.maximum(base,np.ceil(np.maximum(0,low*need/capacity-1e-10)).astype(int))
    # MILP: variables (integer trips, continuous served liters).
    # Cost paid per trip, even if final trip delivers fewer liters.
    lower=np.r_[required,low*need]
    upper=np.r_[max_trips,need]
    c=np.r_[np.zeros(n),-np.ones(n)]
    mat=np.zeros((n+1,2*n))
    mat[:n,:n]=-np.eye(n)*capacity
    mat[:n,n:]=np.eye(n)
    mat[-1,:n]=1
    res=milp(c,integrality=np.r_[np.ones(n),np.zeros(n)],bounds=Bounds(lower,upper),
             constraints=LinearConstraint(mat,np.r_[-np.inf*np.ones(n),-np.inf],
                                           np.r_[np.zeros(n),budget_trips]),
             options={'time_limit':15,'mip_rel_gap':0.001})
    if res.x is None:
        # Feasible integer fallback: fund the required trips only.
        t=required
        return t,np.minimum(need,t*capacity)
    t=np.rint(res.x[:n]).astype(int)
    return t,np.minimum(need,np.maximum(0,res.x[n:]))


def coverage_summary(need, served, trips, trip_cost, budget):
    cov=np.divide(served,need,out=np.ones_like(need),where=need>0)
    spent=float(trips.sum()*trip_cost)
    return {'Total coverage':float(served.sum()/need.sum()) if need.sum()>0 else 1.,
            'Minimum coverage':float(cov.min()),
            'Dana terpakai (Rp)':spent,
            'Dana tidak terpakai (Rp)':float(budget-spent)}

with st.sidebar:
    st.header('⚙️ Parameter simulasi')
    st.success('📊 **Dataset aktif: CHIRPS Gunungkidul 2000–2025**')
    st.caption('Data analisis SPI-3 dan episode kekeringan sudah tersedia secara otomatis.')
    with st.expander('📁 Pengaturan lanjutan: ganti dataset (opsional)', expanded=False):
        st.caption('Hanya diperlukan jika ingin menggunakan data sendiri. Excel wajib memuat sheet Data_Bulanan_SPI dan Episode_Kekeringan.')
        uploaded=st.file_uploader('Unggah Excel pengganti',type=['xlsx'])
    st.divider()
    n_sim=st.select_slider('Jumlah simulasi',options=[1000,3000,5000,10000],value=5000)
    seed=st.number_input('Random seed',min_value=1,value=2026)
    q=st.number_input('Liter/orang/hari (asumsi)',5.,50.,15.,1.)
    capacity=st.number_input('Kapasitas tangki (liter; asumsi)',1000,20000,5000,500)
    trip_cost=st.number_input('Biaya per rit (Rp; asumsi)',50000,2000000,350000,25000)
    bpbd=st.number_input('Dana pusat/BPBD (Rp)',0,10000000000,346000000,10000000)
    local_total=st.number_input('Dana kapanewon total (Rp; asumsi)',0,10000000000,500000000,10000000)
    st.subheader('Durasi bantuan per bulan SPI')
    days_lo=st.slider('Hari minimum',1,30,7)
    days_hi=st.slider('Hari maksimum',days_lo,40,max(14,days_lo))
    st.subheader('Sasaran bantuan (asumsi)')
    target_lo = st.number_input(
        "Proporsi sasaran minimum",
        min_value=0.0, max_value=0.5,
        value=0.025, step=0.005
    )

    target_mode = st.number_input(
        "Proporsi sasaran paling mungkin",
        min_value=0.0, max_value=0.5,
        value=0.07, step=0.005
    )

    target_hi = st.number_input(
        "Proporsi sasaran maksimum",
        min_value=0.0, max_value=0.5,
        value=0.16, step=0.005
    )

    st.subheader("Proporsi kebutuhan dipenuhi dropping air")

    delivery_lo = st.number_input(
        "Proporsi dropping minimum",
        min_value=0.0, max_value=1.0,
        value=0.25, step=0.05
    )

    delivery_mode = st.number_input(
        "Proporsi dropping paling mungkin",
        min_value=0.0, max_value=1.0,
        value=0.50, step=0.05
    )

    delivery_hi = st.number_input(
        "Proporsi dropping maksimum",
        min_value=0.0, max_value=1.0,
        value=0.75, step=0.05
    )

    if not (target_lo <= target_mode <= target_hi):
        st.error("Proporsi sasaran harus minimum ≤ modus ≤ maksimum.")
        st.stop()

    if not (delivery_lo <= delivery_mode <= delivery_hi):
        st.error("Proporsi dropping harus minimum ≤ modus ≤ maksimum.")
        st.stop()
    sigma=st.slider('Heterogenitas sasaran antarwilayah',0.0,1.0,0.5,0.05)
    st.caption('Rp346 juta: angka referensi pemberitaan 2026 yang perlu diverifikasi, bukan total dana bantuan air. Semua nilai lainnya dapat diganti.')

# Data publik contoh bawaan agar pengunjung dapat langsung mencoba dashboard.
# File unggahan pengguna, jika ada, selalu menggantikan data bawaan.
DEFAULT_DATA = Path(__file__).parent / "data" / "KARST_FUND_Analisis_Kekeringan.xlsx"
if uploaded is None and not DEFAULT_DATA.exists():
    st.info('📁 Unggah Excel analisis kekeringan di sidebar untuk mengaktifkan dashboard.')
    st.stop()
if uploaded is None:
    st.caption('✅ **Data historis bawaan aktif.** Kamu dapat langsung menggunakan semua fitur dashboard.')
try:
    raw_data = uploaded.getvalue() if uploaded is not None else DEFAULT_DATA.read_bytes()
    monthly, episodes=read_historical(raw_data)
    if uploaded is not None:
        st.caption('📁 **Dataset unggahan aktif.** Hasil simulasi menggunakan file pengganti yang diunggah.')
    lengths=pd.to_numeric(episodes['Durasi_Bulan'],errors='coerce').dropna()
    lengths=lengths[lengths>0]
    if len(lengths)<3:st.error('Sheet episode harus memuat minimal tiga durasi valid.');st.stop()
except Exception as e:
    st.error(f'Gagal membaca Excel: {e}');st.stop()

if not(target_lo<=target_mode<=target_hi and delivery_lo<=delivery_mode<=delivery_hi):
    st.error('Urutan parameter segitiga harus minimum ≤ modus ≤ maksimum.');st.stop()

budget=bpbd+local_total
try:
    days,need,trips,costs,loss,target,delivery=simulate(tuple(lengths),n_sim,int(seed),q,capacity,trip_cost,days_lo,days_hi,target_lo,target_mode,target_hi,delivery_lo,delivery_mode,delivery_hi,sigma)
except Exception as e:
    st.error(str(e));st.stop()

expected=loss.mean();var95=np.quantile(loss,.95);cvar95=np.sort(loss)[-int(np.ceil(n_sim*.05)):].mean()
# Annual model: 20 observed episodes over 26 years (2000-2025), Poisson frequency.
# A transparent exploratory assumption, NOT validated annual forecast.
annual_rate=len(lengths)/26.0
annual_rng=np.random.default_rng(int(seed)+100000)
annual_counts=annual_rng.poisson(annual_rate,n_sim)
annual_losses=np.zeros(n_sim)
for j,k in enumerate(annual_counts):
    if k:annual_losses[j]=loss[annual_rng.integers(0,n_sim,size=k)].sum()
annual_var95=np.quantile(annual_losses,.95)
annual_tail=annual_losses[annual_losses>=annual_var95]
annual_cvar95=annual_tail.mean() if len(annual_tail) else 0.0


# v8: independent exact integer-trip target calculator (uniform trip cost).
def target_trip_plan(need_liters, target_fraction, tank_liters, cost_per_trip, local_floor=None):
    """Return exact minimum trips for regional minimum coverage target.

    For hybrid, precommitted local trips are mandatory. This is a financial
    feasibility model, not a routing or water-source capacity model.
    """
    n=np.asarray(need_liters,dtype=float)
    if np.any(~np.isfinite(n)) or np.any(n<0):
        raise ValueError('Kebutuhan liter harus finite dan tidak negatif.')
    if tank_liters<=0 or cost_per_trip<=0 or not 0<=target_fraction<=1:
        raise ValueError('Parameter kapasitas, biaya, atau target tidak valid.')
    target=np.ceil(np.maximum(0,n*target_fraction/tank_liters-1e-10)).astype(int)
    cap=np.ceil(n/tank_liters-1e-10).astype(int)
    if local_floor is not None:
        target=np.maximum(target,np.minimum(np.asarray(local_floor,dtype=int),cap))
    target=np.minimum(target,cap)
    served=np.minimum(n,target*tank_liters)
    return target,served,int(target.sum())*int(cost_per_trip)


def max_feasible_target(need_liters, available_budget, tank_liters, cost_per_trip, local_floor=None):
    """Find exact best max-min coverage over integer trips via coverage breakpoints."""
    n=np.asarray(need_liters,dtype=float)
    cap=np.ceil(n/tank_liters-1e-10).astype(int)
    base=np.zeros(len(n),dtype=int) if local_floor is None else np.minimum(np.asarray(local_floor,dtype=int),cap)
    trip_limit=int(np.floor((available_budget+1e-7)/cost_per_trip))
    if base.sum()>trip_limit:
        return None,None,None
    # All attainable coverage thresholds are finite breakpoints k*tank/need.
    # The best max-min threshold is one of these breakpoints.
    breaks=np.unique(np.concatenate(([0.,1.],*[np.minimum(1.,np.arange(1,int(k)+1)*tank_liters/v) for k,v in zip(cap,n) if v>0])))
    lo,hi=0,len(breaks)-1
    while lo<hi:
        mid=(lo+hi+1)//2
        t,_,_=target_trip_plan(n,float(breaks[mid]),tank_liters,cost_per_trip,base)
        if t.sum()<=trip_limit:lo=mid
        else:hi=mid-1
    # Allocate exactly enough to guarantee optimum minimum coverage.
    t,served,spent=target_trip_plan(n,float(breaks[lo]),tank_liters,cost_per_trip,base)
    actual=np.divide(served,n,out=np.ones_like(n),where=n>0)
    return float(actual.min()),t,spent

tabs=st.tabs(['🌧️ Kekeringan historis','💰 Dana cadangan','💧 Optimisasi dana','📊 Stress test','📘 Metodologi','🧪 Audit Monte Carlo','📈 Sensitivitas anggaran','🛟 Aktivasi dana cadangan','🎯 Validasi target'])
with tabs[0]:
    st.subheader('Karakteristik kekeringan meteorologis Gunungkidul')
    c1,c2,c3=st.columns(3)
    c1.metric('Episode SPI historis',len(lengths));c2.metric('Durasi median',f'{lengths.median():g} bulan');c3.metric('Durasi terpanjang',f'{lengths.max():g} bulan')
    spi_cols=[x for x in monthly.columns if 'SPI' in str(x).upper()]
    date_cols=[x for x in monthly.columns if 'tanggal' in str(x).lower() or 'bulan' in str(x).lower()]
    if spi_cols:
        xcol=date_cols[0] if date_cols else monthly.columns[0]
        plot=monthly[[xcol,spi_cols[0]]].copy();plot[xcol]=pd.to_datetime(plot[xcol],errors='coerce')
        fig=px.line(plot,x=xcol,y=spi_cols[0],title=f'{spi_cols[0]} — data historis');fig.add_hline(y=-1,line_dash='dash',annotation_text='SPI = -1')
        st.plotly_chart(fig,use_container_width=True)
    st.dataframe(episodes,use_container_width=True,hide_index=True)
    st.warning('SPI tingkat kabupaten tidak menunjukkan risiko spesifik masing-masing kapanewon; durasi episode SPI bukan durasi dropping air aktual.')
with tabs[1]:
    st.subheader('KARST-RESERVE | Simulasi kebutuhan pembiayaan')
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Rata-rata kebutuhan',f'Rp{expected/1e9:.2f} M')
    c2.metric('VaR 95%',f'Rp{var95/1e9:.2f} M')
    c3.metric('CVaR 95%',f'Rp{cvar95/1e9:.2f} M')
    c4.metric('P(kekurangan)',f'{(loss>budget).mean():.1%}')
    st.caption(f'Anggaran pembanding Rp{budget/1e6:,.0f} juta; rata-rata kekurangan Rp{np.maximum(loss-budget,0).mean()/1e6:,.1f} juta.')
    fig=px.histogram(pd.DataFrame({'Kebutuhan dana (Rp miliar)':loss/1e9}),x='Kebutuhan dana (Rp miliar)',nbins=45,title='Distribusi kebutuhan dana bersyarat pada episode SPI')
    fig.add_vline(x=budget/1e9,line_dash='dash',annotation_text='Anggaran')
    st.plotly_chart(fig,use_container_width=True)
    st.markdown('#### Model eksploratif kebutuhan dana tahunan')
    st.caption(f'Frekuensi episode diasumsikan Poisson dengan rata-rata {annual_rate:.3f} episode/tahun = {len(lengths)} episode / 26 tahun. Asumsi ini belum tervalidasi.')
    a1,a2,a3=st.columns(3)
    a1.metric('Expected tahunan',f'Rp{annual_losses.mean()/1e9:.2f} M')
    a2.metric('VaR 95% tahunan',f'Rp{annual_var95/1e9:.2f} M')
    a3.metric('P(shortfall tahunan)',f'{(annual_losses>budget).mean():.1%}')
    st.caption(f'CVaR 95% tahunan (rata-rata ekor >= kuantil): Rp{annual_cvar95/1e9:.2f} M. Asumsi episode independen dan durasi historis di-bootstrap.')
    grid=np.linspace(0,max(var95*1.3,budget*1.3,1),80)
    curve=pd.DataFrame({'Anggaran (Rp miliar)':grid/1e9,'Probabilitas kekurangan':[(loss>b).mean() for b in grid]})
    st.plotly_chart(px.line(curve,x='Anggaran (Rp miliar)',y='Probabilitas kekurangan',title='Kecukupan dana pada berbagai tingkat anggaran'),use_container_width=True)
    st.download_button('⬇️ Unduh simulasi (CSV)',pd.DataFrame({'Hari_bantuan':days,'Kebutuhan_dana_Rp':loss}).to_csv(index=False).encode(),'karst_reserve_simulasi.csv','text/csv')
with tabs[2]:
    st.subheader('KARST-POOL | Tiga skema, total anggaran identik')
    chosen=st.selectbox('Pilih kebutuhan untuk perbandingan',('Median','VaR 95%','Skenario terbesar'))
    idx={'Median':int(np.argmin(abs(loss-np.median(loss)))),'VaR 95%':int(np.argmin(abs(loss-var95))),'Skenario terbesar':int(np.argmax(loss))}[chosen]
    need_i=need[idx]
    # Separate: entire budget preallocated by population; no interregional transfer.
    fixed=budget*POP/POP.sum()
    sep_trips=np.minimum(np.floor(fixed/trip_cost).astype(int),np.ceil(need_i/capacity).astype(int))
    sep_served=np.minimum(need_i,sep_trips*capacity)
    # Hybrid: local budgets are spent first; remaining central budget flexible.
    local=local_total*POP/POP.sum()
    local_trips=np.minimum(np.floor(local/trip_cost).astype(int),np.ceil(need_i/capacity).astype(int))
    hybrid_trips,hybrid_served=integer_fair_allocate(need_i,budget,capacity,trip_cost,local_trips)
    pooled_trips,pooled_served=integer_fair_allocate(need_i,budget,capacity,trip_cost)
    rows=[];regional=[]
    for name,t,served in [('Separate',sep_trips,sep_served),('Hybrid',hybrid_trips,hybrid_served),('Pooling',pooled_trips,pooled_served)]:
        summary=coverage_summary(need_i,served,t,trip_cost,budget)
        rows.append({'Skema':name,**summary,'Total rit':int(t.sum())})
        cov=np.divide(served,need_i,out=np.ones_like(need_i),where=need_i>0)
        for region,v,c,nt in zip(REGIONS,served,cov,t):
            regional.append({'Kapanewon':region,'Skema':name,'Alokasi liter':v,'Coverage':c,'Jumlah rit':int(nt),'Biaya (Rp)':int(nt)*trip_cost})
    st.dataframe(pd.DataFrame(rows).style.format({'Total coverage':'{:.1%}','Minimum coverage':'{:.1%}','Dana terpakai (Rp)':'{:,.0f}','Dana tidak terpakai (Rp)':'{:,.0f}'}),hide_index=True,use_container_width=True)
    st.plotly_chart(px.bar(pd.DataFrame(regional),x='Kapanewon',y='Coverage',color='Skema',barmode='group',title=f'Tingkat pemenuhan per kapanewon — {chosen}'),use_container_width=True)
    st.download_button('⬇️ Unduh alokasi (CSV)',pd.DataFrame(regional).to_csv(index=False).encode(),'karst_alokasi_v4.csv','text/csv')
    st.info('v4 menggunakan perjalanan tangki bilangan bulat. Separate mengunci seluruh anggaran per wilayah; Hybrid mengunci realisasi perjalanan lokal lalu mengalokasikan sisa dana fleksibel; Pooling mengalokasikan semua dana fleksibel. Anggaran lokal/total adalah asumsi.')

with tabs[3]:
    st.subheader('Analisis sensitivitas dengan sampel acak yang sama')
    cases=[('Baseline',1,1,1),('Sasaran +25%',1.25,1,1),('Sasaran -25%',.75,1,1),('Hari +30%',1,1.3,1),('Hari -20%',1,.8,1),('Biaya +25%',1,1,1.25),('Biaya -15%',1,1,.85),('Gabungan buruk',1.25,1.3,1.25)]
    records=[]
    for name,t,d,c in cases:
        share=np.clip(target*t,0,1)
        vol=POP[None,:]*share*q*(days*d)[:,None]*delivery[:,None]
        l=np.ceil(vol/capacity).sum(axis=1)*trip_cost*c
        records.append({'Skenario':name,'Expected (Rp miliar)':l.mean()/1e9,'VaR95 (Rp miliar)':np.quantile(l,.95)/1e9,'P(kekurangan)':(l>budget).mean()})
    sens=pd.DataFrame(records)
    st.plotly_chart(px.bar(sens,x='Expected (Rp miliar)',y='Skenario',orientation='h',title='Perbandingan rata-rata kebutuhan dana'),use_container_width=True)
    st.dataframe(sens.style.format({'Expected (Rp miliar)':'{:.3f}','VaR95 (Rp miliar)':'{:.3f}','P(kekurangan)':'{:.1%}'}),hide_index=True,use_container_width=True)
    st.caption('Perubahan antarparameter tidak seragam; grafik bukan peringkat sensitivitas elastisitas.')
with tabs[4]:
    st.markdown('''### Dasar dan batasan penelitian
- **Data empiris:** episode kekeringan SPI dari Excel CHIRPS serta populasi BPS 2025 (angka dibulatkan per kapanewon).
- **Asumsi, bukan fakta terukur:** hari bantuan per bulan SPI, distribusi proporsi sasaran, kebutuhan melalui dropping air, kapasitas tangki, biaya per rit, dan dana kapanewon.
- **Reserve:** bootstrap episode historis, kemudian simulasi asumsi kebutuhan air; pembiayaan dihitung dengan `ceil(liter/kapasitas)` per kapanewon.
- **VaR/CVaR episode:** kuantil 95% dan rata-rata 5% hasil simulasi tertinggi; bersyarat pada asumsi. Model tahunan memakai frekuensi Poisson dari jumlah episode/26 tahun, independensi episode, dan bootstrap biaya; masih eksploratif.
- **Separate:** dana tetap per wilayah proporsional penduduk; **Hybrid:** dana lokal terlebih dahulu, lalu dana pusat fleksibel; **Pooling:** dana sepenuhnya fleksibel.
- **Optimisasi:** optimisasi integer dua tahap (maksimum tingkat pemenuhan minimum, kemudian maksimum volume), belum memperhitungkan rute truk dan kapasitas jalan.
- **Cakupan:** bukan prakiraan tahunan atau estimasi anggaran BPBD aktual. Skema hybrid hanyalah penyederhanaan ilustratif dari mekanisme berjenjang.
- **Perbandingan skema:** hanya sah dalam kerangka asumsi anggaran total sama, biaya unit sama, dan kebutuhan skenario sama.

**Langkah validasi lanjutan:** cek standar air minimum, harga/kapasitas tangki, validasi SPI dan hubungan kondisi meteorologis–permintaan bantuan, serta bandingkan dengan data realisasi BPBD bila tersedia.''')


with tabs[5]:
    st.subheader('Audit Monte Carlo | Ketahanan tiga skema pembiayaan')
    st.caption('Menggunakan kebutuhan yang sama, total anggaran identik, dan perjalanan tangki bilangan bulat. Hasil bersifat simulasi, bukan evaluasi kebijakan aktual.')
    c1,c2=st.columns(2)
    n_audit=c1.selectbox('Jumlah skenario untuk audit',[30,60,100,200],index=1)
    budget_factor=c2.selectbox('Faktor anggaran',[0.5,0.75,1.0,1.25,1.5],index=2,format_func=lambda x:f'{x:.0%} dari anggaran sidebar')
    audit_budget=budget*budget_factor
    st.write(f'Anggaran uji: **Rp{audit_budget:,.0f}**; biaya per rit: **Rp{trip_cost:,.0f}**')
    if st.button('▶️ Jalankan audit',type='primary'):
        with st.spinner('Mengoptimalkan alokasi per skenario; bisa memerlukan waktu beberapa menit...'):
            # Evenly spaced deterministic quantiles of simulated episode costs.
            order=np.argsort(loss)
            indices=order[np.linspace(0,n_sim-1,n_audit).astype(int)]
            result=[]
            errors=[]
            for case_no,ix in enumerate(indices,1):
                ni=need[ix]
                fixed_i=audit_budget*POP/POP.sum()
                t_sep=np.minimum(np.floor(fixed_i/trip_cost).astype(int),np.ceil(ni/capacity).astype(int))
                served_sep=np.minimum(ni,t_sep*capacity)
                # Local commitments are capped at the original local fund AND adjusted
                # when stress-test budget is below the original combined budget.
                local_fund=min(local_total*budget_factor,audit_budget)
                local_i=local_fund*POP/POP.sum()
                t_local=np.minimum(np.floor(local_i/trip_cost).astype(int),np.ceil(ni/capacity).astype(int))
                # Integer rounding may leave the total feasible; each local trip paid once.
                try:
                    t_h,s_h=integer_fair_allocate(ni,audit_budget,capacity,trip_cost,t_local)
                    t_p,s_p=integer_fair_allocate(ni,audit_budget,capacity,trip_cost)
                except Exception as exc:
                    errors.append(f'Skenario {case_no}: {exc}')
                    continue
                for name,t,sv in [('Separate',t_sep,served_sep),('Hybrid',t_h,s_h),('Pooling',t_p,s_p)]:
                    row=coverage_summary(ni,sv,t,trip_cost,audit_budget)
                    result.append({'Skenario':case_no,'Biaya kebutuhan penuh (Rp)':float(loss[ix]),'Skema':name,**row,'Total rit':int(t.sum())})
                    if t.sum()*trip_cost>audit_budget+0.01 or np.any(sv>ni+1e-5) or np.any(sv< -1e-5):
                        errors.append(f'Pelanggaran kendala pada skenario {case_no}, {name}')
            audit_df=pd.DataFrame(result)
            st.session_state['karst_audit_df']=audit_df
            st.session_state['karst_audit_errors']=errors
            st.session_state['karst_audit_meta']=(n_audit,budget_factor,seed)
    if 'karst_audit_df' in st.session_state:
        a=st.session_state['karst_audit_df']
        if a.empty:
            st.error('Tidak ada skenario audit yang berhasil dijalankan.')
        else:
            summary=a.groupby('Skema',as_index=False).agg(
                Rata_rata_total_coverage=('Total coverage','mean'),
                Rata_rata_minimum_coverage=('Minimum coverage','mean'),
                Rata_rata_dana_menganggur_Rp=('Dana tidak terpakai (Rp)','mean'),
                Rata_rata_rit=('Total rit','mean'))
            st.markdown('#### Ringkasan hasil')
            st.dataframe(summary.style.format({'Rata_rata_total_coverage':'{:.1%}',
                'Rata_rata_minimum_coverage':'{:.1%}',
                'Rata_rata_dana_menganggur_Rp':'{:,.0f}',
                'Rata_rata_rit':'{:,.1f}'}),hide_index=True,use_container_width=True)
            pivot=a.pivot(index='Skenario',columns='Skema',values='Minimum coverage')
            if {'Pooling','Hybrid','Separate'}.issubset(pivot.columns):
                diff_ph=pivot['Pooling']-pivot['Hybrid']
                diff_ps=pivot['Pooling']-pivot['Separate']
                d1,d2,d3=st.columns(3)
                d1.metric('Pooling unggul dari Hybrid',f'{(diff_ph>1e-6).mean():.1%} skenario')
                d2.metric('Rata-rata selisih minimum coverage',f'{diff_ph.mean()*100:+.2f} pp')
                d3.metric('Pooling unggul dari Separate',f'{(diff_ps>1e-6).mean():.1%} skenario')
                st.plotly_chart(px.histogram(pd.DataFrame({'Selisih minimum coverage (poin persentase)':diff_ph*100}),
                    x='Selisih minimum coverage (poin persentase)',nbins=25,
                    title='Pooling minus Hybrid: pemerataan per skenario'),use_container_width=True)
            st.download_button('⬇️ Unduh hasil audit (CSV)',a.to_csv(index=False).encode('utf-8-sig'),
                'KARST_FUND_v5_Audit_Monte_Carlo.csv','text/csv')
            if st.session_state.get('karst_audit_errors'):
                st.error('Temuan audit: '+str(st.session_state['karst_audit_errors'][:8]))
            else:
                st.success('Tidak ditemukan pelanggaran anggaran/volume pada skenario yang berhasil diuji.')
            st.warning('Catatan: skenario diambil berdasarkan kuantil biaya dari simulasi, bukan sampel acak representatif untuk menghitung peluang populasi. Persentase kemenangan adalah proporsi skenario uji, bukan probabilitas kebijakan. Hasil uji juga bergantung pada input sidebar ketika tombol ditekan.')


with tabs[6]:
    st.subheader('Sensitivitas anggaran | Kapan pooling memberikan manfaat?')
    st.caption('Kebutuhan skenario identik di semua tingkat anggaran. Dana lokal dan dana pusat sama-sama dikalikan faktor anggaran agar komposisinya konsisten.')
    factors=[0.5,0.75,1.0,1.25,1.5]
    n_budget=st.selectbox('Jumlah skenario kebutuhan per tingkat anggaran',[15,30,60],index=1,
                          help='Semakin banyak skenario, semakin lama optimisasi integer.')
    st.dataframe(pd.DataFrame({'Faktor anggaran':[f'{x:.0%}' for x in factors],
                               'Total dana (Rp)':[round(budget*x) for x in factors],
                               'Dana lokal (Rp)':[round(local_total*x) for x in factors],
                               'Dana pusat (Rp)':[round(bpbd*x) for x in factors]}),
                 hide_index=True,use_container_width=True)
    st.info('Gunakan 15 skenario dahulu untuk mengecek dashboard. 30 skenario untuk analisis awal; 60 untuk uji ketahanan lebih luas.')
    if st.button('▶️ Jalankan sensitivitas anggaran',type='primary'):
        with st.spinner('Menghitung lima tingkat anggaran × tiga skema. Proses dapat memakan waktu beberapa menit...'):
            selected=np.argsort(loss)[np.linspace(0,n_sim-1,n_budget).astype(int)]
            rows=[]; errors=[]
            for factor in factors:
                b=float(budget*factor)
                local_b=float(local_total*factor)
                for j,ix in enumerate(selected,1):
                    ni=need[ix]
                    max_t=np.ceil(ni/capacity).astype(int)
                    separate_i=b*POP/POP.sum()
                    sep=np.minimum(np.floor(separate_i/trip_cost).astype(int),max_t)
                    loc_i=local_b*POP/POP.sum()
                    local_t=np.minimum(np.floor(loc_i/trip_cost).astype(int),max_t)
                    try:
                        hy,served_hy=integer_fair_allocate(ni,b,capacity,trip_cost,local_t)
                        po,served_po=integer_fair_allocate(ni,b,capacity,trip_cost)
                    except Exception as exc:
                        errors.append(f'Faktor {factor:.0%}, skenario {j}: {exc}')
                        continue
                    for scheme,t,sv in [('Separate',sep,np.minimum(ni,sep*capacity)),
                                        ('Hybrid',hy,served_hy),('Pooling',po,served_po)]:
                        met=coverage_summary(ni,sv,t,trip_cost,b)
                        rows.append({'Faktor':factor,'Anggaran (Rp)':b,'Skenario':j,
                                     'Kebutuhan penuh (Rp)':float(loss[ix]),'Skema':scheme,
                                     **met,'Total rit':int(t.sum())})
                        if t.sum()*trip_cost>b+0.01 or np.any(sv>ni+1e-5) or np.any(sv < -1e-5):
                            errors.append(f'Pelanggaran kendala {factor:.0%} skenario {j} {scheme}')
            st.session_state['karst_v6_results']=pd.DataFrame(rows)
            st.session_state['karst_v6_errors']=errors
            st.session_state['karst_v6_signature']=(n_budget,seed,budget,local_total,trip_cost,capacity,n_sim)
    if 'karst_v6_results' in st.session_state:
        df=st.session_state['karst_v6_results']
        sig=(n_budget,seed,budget,local_total,trip_cost,capacity,n_sim)
        if st.session_state.get('karst_v6_signature')!=sig:
            st.warning('Parameter telah berubah sejak analisis terakhir. Klik Jalankan sensitivitas anggaran lagi untuk memperbarui hasil.')
        if df.empty:
            st.error('Belum ada skenario yang berhasil dihitung.')
        else:
            group=df.groupby(['Faktor','Skema'],as_index=False).agg(
                Total_coverage=('Total coverage','mean'),Minimum_coverage=('Minimum coverage','mean'),
                Dana_tersisa=('Dana tidak terpakai (Rp)','mean'),Rit=('Total rit','mean'))
            st.markdown('#### Rata-rata indikator menurut anggaran')
            st.dataframe(group.style.format({'Faktor':'{:.0%}','Total_coverage':'{:.1%}',
                'Minimum_coverage':'{:.1%}','Dana_tersisa':'{:,.0f}','Rit':'{:,.1f}'}),
                hide_index=True,use_container_width=True)
            st.plotly_chart(px.line(group,x='Faktor',y='Minimum_coverage',color='Skema',markers=True,
                labels={'Faktor':'Faktor anggaran','Minimum_coverage':'Rata-rata minimum coverage'},
                title='Pemerataan bantuan pada lima tingkat anggaran'),use_container_width=True)
            st.plotly_chart(px.line(group,x='Faktor',y='Total_coverage',color='Skema',markers=True,
                labels={'Faktor':'Faktor anggaran','Total_coverage':'Rata-rata total coverage'},
                title='Cakupan total pada lima tingkat anggaran'),use_container_width=True)
            pivot=df.pivot_table(index=['Faktor','Skenario'],columns='Skema',values='Minimum coverage')
            if {'Pooling','Hybrid','Separate'}.issubset(pivot.columns):
                differences=pivot.assign(
                    delta_pool_hybrid_pp=(pivot['Pooling']-pivot['Hybrid'])*100,
                    delta_pool_separate_pp=(pivot['Pooling']-pivot['Separate'])*100).reset_index()
                gains=differences.groupby('Faktor',as_index=False).agg(
                    Delta_Pooling_Hybrid_pp=('delta_pool_hybrid_pp','mean'),
                    Delta_Pooling_Separate_pp=('delta_pool_separate_pp','mean'),
                    Pooling_unggul_Hybrid=('delta_pool_hybrid_pp',lambda x: (x>1e-6).mean()),
                    Pooling_unggul_Separate=('delta_pool_separate_pp',lambda x: (x>1e-6).mean()))
                st.markdown('#### Manfaat tambahan pooling (minimum coverage)')
                st.dataframe(gains.style.format({'Faktor':'{:.0%}',
                    'Delta_Pooling_Hybrid_pp':'{:+.2f}', 'Delta_Pooling_Separate_pp':'{:+.2f}',
                    'Pooling_unggul_Hybrid':'{:.1%}','Pooling_unggul_Separate':'{:.1%}'}),
                    hide_index=True,use_container_width=True)
                st.plotly_chart(px.bar(gains,x='Faktor',y='Delta_Pooling_Hybrid_pp',
                    labels={'Faktor':'Faktor anggaran','Delta_Pooling_Hybrid_pp':'Selisih minimum coverage (pp)'},
                    title='Tambahan pemerataan Pooling dibanding Hybrid'),use_container_width=True)
            st.download_button('⬇️ Unduh sensitivitas anggaran (CSV)',
                df.to_csv(index=False).encode('utf-8-sig'),
                'KARST_FUND_v6_Sensitivitas_Anggaran.csv','text/csv')
            if st.session_state.get('karst_v6_errors'):
                st.error('Ada kendala: '+str(st.session_state['karst_v6_errors'][:8]))
            else:
                st.success('Seluruh hasil yang dihitung lolos pemeriksaan dasar anggaran dan volume.')
            st.warning('Skenario dipilih menurut kuantil kebutuhan, sehingga rata-ratanya merupakan rata-rata skenario uji, bukan ekspektasi probabilistik tahunan. Faktor anggaran merupakan eksperimen hipotetis. Kinerja pooling relatif terhadap hybrid sebagian dipengaruhi aturan optimisasi dan komitmen lokal.')


with tabs[7]:
    st.subheader('KARST-RESERVE | Simulasi aktivasi dana cadangan')
    st.caption('Rekomendasi kebutuhan tambahan dana untuk mencapai target cakupan minimum setiap kapanewon pada SATU skenario kekeringan. Ini bukan prediksi kebutuhan resmi atau perintah transfer anggaran.')
    col_a,col_b=st.columns(2)
    with col_a:
        reserve_case=st.selectbox('Skenario kebutuhan bantuan',('Median','VaR 95%','Skenario terbesar'),key='reserve_case')
        coverage_target=st.slider('Target minimum cakupan tiap kapanewon',min_value=0,max_value=100,value=80,step=5,format='%d%%') / 100.0
    with col_b:
        mechanism=st.selectbox('Mekanisme alokasi',('Pooling fleksibel','Hybrid: perjalanan lokal dipertahankan'),key='reserve_mechanism')
        available_extra=st.number_input('Dana cadangan yang dapat diaktifkan (Rp; asumsi)',min_value=0,max_value=10000000000,value=300000000,step=10000000)
    idx_r={'Median':int(np.argmin(abs(loss-np.median(loss)))),
           'VaR 95%':int(np.argmin(abs(loss-var95))),
           'Skenario terbesar':int(np.argmax(loss))}[reserve_case]
    need_r=np.asarray(need[idx_r],dtype=float)
    # One truck is indivisible; ceil ensures every region can meet the requested minimum.
    target_trips=np.ceil(np.maximum(0,coverage_target*need_r/capacity-1e-12)).astype(int)
    if mechanism.startswith('Hybrid'):
        local_r=local_total*POP/POP.sum()
        local_commit=np.minimum(np.floor(local_r/trip_cost).astype(int),np.ceil(need_r/capacity).astype(int))
    else:
        local_commit=np.zeros(len(POP),dtype=int)
    required_trips=np.maximum(target_trips,local_commit)
    minimum_cost=int(required_trips.sum())*int(trip_cost)
    # Actual pre-funded trips are whole numbers; budget available for new trips is floored.
    current_capacity=int(np.floor(budget/trip_cost))
    required_budget_trips=int(required_trips.sum())
    shortfall_trips=max(0,required_budget_trips-current_capacity)
    additional_min=shortfall_trips*int(trip_cost)
    st.markdown('#### Rekomendasi dana untuk target')
    r1,r2,r3,r4=st.columns(4)
    r1.metric('Biaya minimum target',f'Rp{minimum_cost/1e6:,.2f} juta')
    r2.metric('Dana tersedia',f'Rp{budget/1e6:,.2f} juta')
    r3.metric('Tambahan minimum (rit bulat)',f'Rp{additional_min/1e6:,.2f} juta')
    r4.metric('Status dana cadangan','CUKUP' if available_extra>=additional_min else 'BELUM CUKUP')
    if additional_min>0:
        st.warning(f'Untuk mencapai target {coverage_target:.0%}, model memerlukan sedikitnya {shortfall_trips:,} rit tambahan (Rp{additional_min:,.0f}). Dana cadangan yang dimasukkan {"mencukupi" if available_extra>=additional_min else "belum mencukupi"}.')
    else:
        st.success('Anggaran awal sudah mencukupi target cakupan minimum pada skenario ini, berdasarkan asumsi biaya dan kebutuhan model.')
    # Compare current optimal allocation to allocation with activated reserve.
    try:
        base_alloc,base_served=integer_fair_allocate(need_r,budget,capacity,trip_cost,local_commit)
        activated_alloc,activated_served=integer_fair_allocate(need_r,budget+available_extra,capacity,trip_cost,local_commit)
        m0=coverage_summary(need_r,base_served,base_alloc,trip_cost,budget)
        m1=coverage_summary(need_r,activated_served,activated_alloc,trip_cost,budget+available_extra)
        st.markdown('#### Dampak aktivasi cadangan terhadap cakupan')
        comp=pd.DataFrame([{'Kondisi':'Tanpa aktivasi','Total coverage':m0['Total coverage'],'Minimum coverage':m0['Minimum coverage'],'Rit':int(base_alloc.sum()),'Dana terpakai (Rp)':m0['Dana terpakai (Rp)']},
                           {'Kondisi':'Dengan aktivasi','Total coverage':m1['Total coverage'],'Minimum coverage':m1['Minimum coverage'],'Rit':int(activated_alloc.sum()),'Dana terpakai (Rp)':m1['Dana terpakai (Rp)']}])
        st.dataframe(comp.style.format({'Total coverage':'{:.1%}','Minimum coverage':'{:.1%}','Dana terpakai (Rp)':'{:,.0f}'}),hide_index=True,use_container_width=True)
        details=pd.DataFrame({'Kapanewon':REGIONS,'Kebutuhan liter':need_r,'Rit minimum untuk target':target_trips,
                              'Rit lokal wajib':local_commit,'Rit rekomendasi minimum':required_trips,
                              'Coverage tanpa cadangan':np.divide(base_served,need_r,out=np.ones_like(need_r),where=need_r>0),
                              'Coverage dengan cadangan':np.divide(activated_served,need_r,out=np.ones_like(need_r),where=need_r>0)})
        st.download_button('⬇️ Unduh rekomendasi per kapanewon (CSV)',details.to_csv(index=False).encode('utf-8-sig'),'KARST_FUND_v7_Reserve_Activation.csv','text/csv')
    except Exception as exc:
        st.error(f'Gagal menghitung alokasi: {exc}')

    st.markdown('---')
    st.subheader('Cadangan tahunan | Target keandalan pendanaan')
    st.caption('BERBEDA dari target cakupan per episode di atas. Angka berikut menggunakan model frekuensi Poisson eksploratif dan distribusi biaya episode berbasis asumsi; bukan estimasi tervalidasi.')
    reliability=st.select_slider('Target peluang dana tahunan mencukupi',options=[0.80,0.90,0.95,0.99],value=0.95,format_func=lambda x:f'{x:.0%}')
    # For a fixed budget, smallest additional cash to cover the integer-trip annual-loss quantile.
    annual_required=float(np.quantile(annual_losses,reliability,method='higher'))
    annual_extra=max(0.0,annual_required-budget)
    annual_extra_trips=int(np.ceil(annual_extra/trip_cost-1e-12))
    annual_extra_rounded=annual_extra_trips*trip_cost
    y1,y2,y3=st.columns(3)
    y1.metric(f'Kebutuhan dana tahunan P{reliability:.0%}',f'Rp{annual_required/1e6:,.2f} juta')
    y2.metric('Tambahan cadangan tahunan',f'Rp{annual_extra_rounded/1e6:,.2f} juta')
    y3.metric('Peluang dana awal mencukupi',f'{(annual_losses<=budget).mean():.1%}')
    st.info('Catatan: dana tambahan untuk target cakupan satu episode dan cadangan untuk keandalan tahunan adalah dua ukuran berbeda. Jangan dijumlahkan begitu saja. Model belum mencakup akses jalan, kapasitas armada, harga berbeda per lokasi, dan realisasi anggaran pemerintah.')


with tabs[8]:
    st.subheader('KARST-FUND v8 | Validasi target pelayanan dan kecukupan dana')
    st.caption('Analisis pada SATU skenario kebutuhan. Target berarti cakupan minimum setiap kapanewon, bukan persentase populasi yang terdampak.')
    a,b=st.columns(2)
    with a:
        scenario_v8=st.selectbox('Skenario kebutuhan untuk validasi',('Median','VaR 95%','Skenario terbesar'),index=1,key='v8_scenario')
        scheme_v8=st.selectbox('Aturan pendanaan',('Pooling fleksibel','Hybrid: perjalanan lokal dipertahankan'),key='v8_scheme')
    with b:
        reserve_v8=st.number_input('Dana cadangan tersedia (Rp)',min_value=0,max_value=10000000000,value=300000000,step=10000000,key='v8_reserve')
        desired_v8=st.slider('Target pelayanan untuk validasi',0,100,80,5,format='%d%%',key='v8_target')/100
    ix8={'Median':int(np.argmin(abs(loss-np.median(loss)))),
         'VaR 95%':int(np.argmin(abs(loss-var95))),
         'Skenario terbesar':int(np.argmax(loss))}[scenario_v8]
    need8=np.asarray(need[ix8],dtype=float)
    max8=np.ceil(need8/capacity-1e-10).astype(int)
    if scheme_v8.startswith('Hybrid'):
        local8=local_total*POP/POP.sum()
        local_floor8=np.minimum(np.floor(local8/trip_cost).astype(int),max8)
    else:
        local_floor8=np.zeros(len(POP),dtype=int)
    total8=budget+reserve_v8
    requested_trips8,served8,min_cost8=target_trip_plan(need8,desired_v8,capacity,trip_cost,local_floor8)
    # The current budget can purchase only complete trips.
    base_trip_count8=int(np.floor((budget+1e-7)/trip_cost))
    required_trip_count8=int(requested_trips8.sum())
    extra_trip_count8=max(0,required_trip_count8-base_trip_count8)
    extra_cash8=max(0, min_cost8-budget)
    extra_rit_cost8=extra_trip_count8*trip_cost
    attainable8,attain_trips8,attain_cost8=max_feasible_target(need8,total8,capacity,trip_cost,local_floor8)
    feasible8=min_cost8<=total8+1e-7
    m1,m2,m3,m4=st.columns(4)
    m1.metric('Biaya minimum untuk target',f'Rp{min_cost8/1e6:,.2f} juta')
    m2.metric('Anggaran termasuk cadangan',f'Rp{total8/1e6:,.2f} juta')
    m3.metric('Gap dari anggaran awal (sebelum cadangan)',f'Rp{extra_cash8/1e6:,.2f} juta')
    m4.metric('Target maksimum dengan dana tersedia',f'{attainable8:.2%}' if attainable8 is not None else 'Tidak layak')
    st.info(f'Biaya rit tambahan dari anggaran awal: {extra_trip_count8:,} rit × Rp{trip_cost:,.0f} = Rp{extra_rit_cost8:,.0f}. Kekurangan kas di atas menghitung selisih biaya target dan anggaran awal; keduanya bisa berbeda karena sisa dana tidak cukup membeli satu rit.')
    if feasible8:
        st.success(f'Target {desired_v8:.0%} layak secara finansial menurut model dan anggaran yang dimasukkan.')
    else:
        st.warning(f'Target {desired_v8:.0%} belum layak dengan anggaran total saat ini. Tambahan kas setelah cadangan: Rp{max(0,min_cost8-total8):,.0f}.')
    st.markdown('#### Kurva kebutuhan dana untuk target 0%–100%')
    target_grid=np.arange(0,101,5)
    sensitivity8=[]
    for pct in target_grid:
        t,sv,required=target_trip_plan(need8,pct/100,capacity,trip_cost,local_floor8)
        sensitivity8.append({'Target (%)':int(pct),'Biaya minimum (Rp)':required,
                             'Rit minimum':int(t.sum()),'Gap terhadap anggaran total (Rp)':max(0,required-total8),
                             'Layak':required<=total8+1e-7})
    curve8=pd.DataFrame(sensitivity8)
    fig8=px.line(curve8,x='Target (%)',y='Biaya minimum (Rp)',markers=True,title='Biaya minimum vs target cakupan minimum')
    fig8.add_hline(y=total8,line_dash='dash',annotation_text='Anggaran + cadangan')
    st.plotly_chart(fig8,use_container_width=True)
    st.dataframe(curve8.style.format({'Biaya minimum (Rp)':'{:,.0f}','Gap terhadap anggaran total (Rp)':'{:,.0f}'}),use_container_width=True,hide_index=True)
    st.download_button('⬇️ Unduh sensitivitas target (CSV)',curve8.to_csv(index=False).encode('utf-8-sig'),'KARST_FUND_v8_Sensitivitas_Target.csv','text/csv')
    st.markdown('#### Audit kelayakan dan kebutuhan per kapanewon')
    actual8=np.divide(served8,need8,out=np.ones_like(need8),where=need8>0)
    detail8=pd.DataFrame({'Kapanewon':REGIONS,'Kebutuhan air (liter)':need8,
                          'Rit minimum target':requested_trips8,'Rit komitmen lokal':local_floor8,
                          'Liter terlayani pada target':served8,'Cakupan tercapai':actual8,
                          'Biaya minimum wilayah (Rp)':requested_trips8*trip_cost})
    checks8={
        'Jumlah rit bilangan bulat':bool(np.all(requested_trips8==requested_trips8.astype(int))),
        'Cakupan tiap wilayah mencapai target':bool(np.all(actual8>=desired_v8-1e-9)),
        'Volume dilayani tidak melebihi kebutuhan':bool(np.all(served8<=need8+1e-6)),
        'Biaya = jumlah rit × biaya per rit':bool(min_cost8==requested_trips8.sum()*trip_cost),
        'Kurva biaya tidak menurun saat target naik':bool(np.all(np.diff(curve8['Biaya minimum (Rp)'])>=0)),
        
    }
    st.dataframe(pd.DataFrame([{'Pemeriksaan':k,'Hasil':'LULUS' if v else 'TIDAK LULUS'} for k,v in checks8.items()]),hide_index=True,use_container_width=True)
    st.info('Status kelayakan finansial: ' + ('LAYAK' if feasible8 else 'BELUM LAYAK') + ' berdasarkan anggaran termasuk cadangan.')
    st.dataframe(detail8.style.format({'Kebutuhan air (liter)':'{:,.0f}',
        'Liter terlayani pada target':'{:,.0f}','Cakupan tercapai':'{:.2%}',
        'Biaya minimum wilayah (Rp)':'{:,.0f}'}),hide_index=True,use_container_width=True)
    st.download_button('⬇️ Unduh validasi per kapanewon (CSV)',detail8.to_csv(index=False).encode('utf-8-sig'),'KARST_FUND_v8_Validasi_Kapanewon.csv','text/csv')
    st.warning('Batasan: biaya per rit diasumsikan seragam; tidak ada batas armada, sumber air, waktu pengiriman, atau akses jalan. Target maksimum dihitung persis dalam model perjalanan bulat, tetapi bukan jaminan kelayakan lapangan. Model kebutuhan air tetap menggunakan asumsi dan SPI tingkat kabupaten.')
