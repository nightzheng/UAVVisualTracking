import { createApp, computed, ref, watch } from "vue/dist/vue.esm-bundler.js";
import "./style.css";

const arms = {
  A: "A · 反应式 / 固定站位", B: "B · CV / 固定站位",
  C: "C · 反应式 / 本体—云台联合", D: "D · CV / 本体—云台联合",
};
const finalStates = ["completed", "failed", "cancelled"];

async function api(path, data) {
  const response = await fetch(path, data === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || response.statusText);
  return body;
}

createApp({
  setup() {
    const presets = ref({});
    const config = ref(null);
    const sampled = ref(null);
    const configError = ref("");
    const taskError = ref("");
    const jobs = ref([]);
    const selected = ref("");
    const logs = ref({});
    const result = ref(null);
    const proof = ref([]);
    const currentFrame = ref(null);
    const comparison = ref(null);
    const comparedId = ref("");
    const drag = ref(null);
    const svg = ref(null);
    const video = ref(null);
    const points = computed(() => config.value?.rotor_motion.omega_points_deg_s || []);
    const active = computed(() => jobs.value.some(job => !finalStates.includes(job.state)));
    const job = computed(() => jobs.value.find(item => item.id === selected.value));
    const base = computed(() => "/api/jobs/" + selected.value);
    const x = t => 40 + 690 * t / (config.value.duration_s || 72);
    const y = speed => 117 - 65 * speed;
    const edited = computed(() => points.value.map(([t, speed]) => [x(t), y(speed)].join(",")).join(" "));
    const phase = computed(() => sampled.value?.time_s.map((t, i) =>
      [x(t), 117 - 5 * sampled.value.phase_deg[i]].join(",")).join(" "));
    function load(scene, seeded = false, mode = "development") {
      config.value = JSON.parse(JSON.stringify(presets.value[scene][seeded ? 1 : 0]));
      config.value.mode = mode;
    }
    function changeMode(mode) {
      if (mode === "development") config.value.mode = mode;
      else {
        const arm = config.value.arm;
        load(config.value.scene, false, mode);
        config.value.arm = arm;
      }
    }
    let previewTimer;
    let revision = 0;
    watch(config, () => {
      clearTimeout(previewTimer);
      const version = ++revision;
      sampled.value = null;
      previewTimer = setTimeout(async () => {
        try {
          const response = await api("/api/preview", config.value);
          if (version === revision) {
            sampled.value = response;
            configError.value = "";
          }
        } catch (error) {
          if (version === revision) configError.value = error.message;
        }
      }, 110);
    }, { deep: true });

    function position(event) {
      const point = new DOMPoint(event.clientX, event.clientY)
        .matrixTransform(svg.value.getScreenCTM().inverse());
      return [Number(((point.x - 40) / 690 * config.value.duration_s).toFixed(3)),
        Number(((117 - point.y) / 65).toFixed(3))];
    }
    function add(event) {
      if (config.value.mode !== "development") return;
      const [t, speed] = event ? position(event) : [config.value.duration_s / 2, 0];
      points.value.push([t, speed]);
      points.value.sort((a, b) => a[0] - b[0]);
    }
    function startDrag(event, i) {
      if (config.value.mode !== "development") return;
      drag.value = i;
      svg.value.setPointerCapture(event.pointerId);
    }
    function move(event) {
      const i = drag.value;
      if (i === null) return;
      const [t, speed] = position(event);
      const time = i === 0 || i === points.value.length - 1 ? points.value[i][0] : t;
      points.value[i] = [time, speed];
    }
    async function importConfig(event) {
      try {
        const imported = JSON.parse(await event.target.files[0].text());
        await api("/api/preview", imported);
        config.value = imported;
        taskError.value = "";
      } catch (error) {
        taskError.value = error.message;
      } finally {
        event.target.value = "";
      }
    }
    function download() {
      const link = document.createElement("a");
      link.href = URL.createObjectURL(new Blob([JSON.stringify(config.value, null, 2)],
        { type: "application/json" }));
      link.download = "experiment.json";
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    }

    let refreshing = false;
    let loaded = "";
    async function refresh() {
      if (refreshing) return;
      refreshing = true;
      try {
        jobs.value = await api("/api/jobs");
        if (!selected.value && jobs.value.length) selected.value = jobs.value[0].id;
        const id = selected.value;
        if (!id) return;
        const url = base.value;
        const state = job.value?.state;
        const key = id + state;
        const nextLogs = await api(url + "/logs");
        if (id === selected.value) logs.value = nextLogs;
        if (key !== loaded) {
          const output = await fetch(url + "/result.json");
          if (output.ok && id === selected.value) result.value = await output.json();
          if (state === "completed") {
            const frames = await api(url + "/demo.frames.json");
            if (id === selected.value) proof.value = frames;
          }
          if (id === selected.value) loaded = key;
        }
      } catch (error) {
        taskError.value = error.message;
      } finally {
        refreshing = false;
      }
    }
    watch(selected, () => {
      loaded = "";
      result.value = null;
      proof.value = [];
      currentFrame.value = null;
      logs.value = {};
      comparedId.value = "";
      comparison.value = null;
      refresh();
    });
    watch(comparedId, async id => {
      comparison.value = null;
      if (!id) return;
      try {
        const other = await api("/api/jobs/" + id + "/result.json");
        if (comparedId.value === id) comparison.value = other;
      } catch (error) { taskError.value = error.message; }
    });
    async function run() {
      try {
        taskError.value = "";
        selected.value = (await api("/api/jobs", config.value)).id;
        await refresh();
      } catch (error) {
        taskError.value = error.message;
      }
    }
    async function cancel() {
      try { await api(base.value + "/cancel", {}); }
      catch (error) { taskError.value = error.message; }
    }
    function seek() {
      const index = Math.floor(video.value.currentTime * 20);
      const frame = proof.value[index];
      currentFrame.value = frame?.D || null;
    }
    api("/api/presets").then(data => {
      presets.value = data;
      load("edge_sway");
      refresh();
      setInterval(refresh, 1200);
    }).catch(error => { taskError.value = error.message; });

    return {
      arms, config, configError, taskError, jobs, selected, logs, result, currentFrame, svg, video,
      comparison, comparedId, points, active, job, base, x, y, edited, phase, sampled, load, changeMode, add, startDrag,
      move, endDrag: () => { drag.value = null; }, importConfig, download, run, cancel, seek,
      isTerminal: state => finalStates.includes(state),
      percent: fraction => (100 * fraction).toFixed(2) + "%",
    };
  },
  template: `
    <header><span>UAV VISUAL TRACKING / U4</span><h1>本地巡检实验工作台</h1>
      <p>真实 Gazebo 采集 · C1 条件化材料绑定 · 独立覆盖评价</p></header>
    <main>
      <p v-if="taskError" role="alert" class="error">{{ taskError }}</p>
      <section v-if="config">
        <h2>01 实验配置</h2>
        <div class="controls">
          <label>实验类型<select :value="config.mode" @change="changeMode($event.target.value)">
            <option value="development">开发 · 新采集</option><option value="formal">冻结参数 · 新采集</option>
            <option value="replay">冻结历史 · 回放</option></select></label>
          <label>场景<select :value="config.scene" @change="load($event.target.value, false, config.mode)">
            <option value="edge_sway">低速往复</option><option value="slow_transition">减速过渡</option>
          </select></label>
          <label>策略<select v-model="config.arm"><option v-for="(name, id) in arms" :value="id">{{ name }}</option></select></label>
          <label>时长 (s)<input type="number" v-model.number="config.duration_s" :disabled="config.mode !== 'development'"></label>
        </div>
        <p class="muted">冻结参数包括 0° 初始化、A 片吸力面、12 个材料区域、640×480 相机、20 Hz 采集 / 10 Hz 消费。
          CV 策略展示 0.2 / 0.5 秒合法预测。结果根目录由服务启动参数设置。</p>
        <h3>转子角速度</h3>
        <p class="muted">折线为连续角速度，虚线为积分相位。双击添加，拖动圆点编辑。理想阶跃不在此协议内。</p>
        <svg ref="svg" viewBox="0 0 760 240" @dblclick.self="add($event)"
             @pointermove="move" @pointerup="endDrag" @lostpointercapture="endDrag"
             aria-label="角速度和积分相位曲线">
          <line v-for="speed in [-1.5,0,1.5]" x1="40" x2="730" :y1="y(speed)" :y2="y(speed)" class="grid"/>
          <text x="6" y="23">+1.5</text><text x="16" y="121">0</text><text x="8" y="218">−1.5</text>
          <text x="40" y="232">0 s</text><text x="688" y="232">{{config.duration_s}} s</text>
          <polyline v-if="phase" :points="phase" class="phase"/>
          <polyline :points="edited" class="omega"/>
          <circle v-for="([t,w],i) in points" :cx="x(t)" :cy="y(w)" r="6" @pointerdown="startDrag($event,i)"/>
        </svg>
        <p class="muted">速度 ±1.5 deg/s，斜率 ±0.5 deg/s²，积分相位 ±15°。当前峰值斜率
          {{ sampled ? Math.max(...sampled.acceleration_deg_s2.map(Math.abs)).toFixed(3) + ' deg/s²' : '—' }}。</p>
        <p v-if="configError" class="error" role="alert">{{ configError }}</p>
        <details><summary>关键点表格</summary>
          <table><thead><tr><th>时间 (s)</th><th>角速度 (deg/s)</th><th></th></tr></thead><tbody>
            <tr v-for="(point,i) in points" :key="i">
              <td><input type="number" step="any" v-model.number="point[0]" :disabled="config.mode !== 'development'" :aria-label="'时间 '+i"></td>
              <td><input type="number" step="any" v-model.number="point[1]" :disabled="config.mode !== 'development'" :aria-label="'速度 '+i"></td>
              <td><button :disabled="config.mode !== 'development' || points.length <= 2" @click="points.splice(i,1)">删除</button></td>
            </tr></tbody></table>
          <button :disabled="config.mode !== 'development'" @click="add()">添加关键点</button>
        </details>
        <div class="controls actions">
          <button :disabled="config.mode !== 'development'" @click="load(config.scene,true)">固定种子扰动</button>
          <label class="upload">导入配置<input type="file" accept=".json" @change="importConfig"></label>
          <button :disabled="!sampled" @click="download">保存 / 导出配置</button>
          <button class="primary" :disabled="active || !sampled" @click="run">运行实验</button>
        </div>
      </section>
      <section>
        <h2>02 执行记录</h2>
        <select v-model="selected" aria-label="实验记录"><option disabled value="">暂无实验</option>
          <option v-for="item in jobs" :value="item.id">{{item.id}} · {{item.state}}</option></select>
        <template v-if="job">
          <p>当前阶段 <strong>{{ job.state }}</strong>
            <button v-if="!isTerminal(job.state)" @click="cancel">取消</button></p>
          <p v-if="job.error" class="error">{{job.error}}</p>
          <details open><summary>运行日志</summary><pre v-for="(lines,name) in logs">{{name}}\n{{lines}}</pre></details>
        </template>
      </section>
      <section v-if="result">
        <h2>03 实验结果</h2>
        <p class="source">{{ result.source === 'replay' ? '历史冻结采集 · 独立重算' : '本次实际采集 · 独立评价' }}</p>
        <div class="stats">
          <div><b>{{result.evaluation.region_count}} / {{result.regions.length}}</b>完成区域</div>
          <div><b>{{percent(result.evaluation.coverage)}}</b>独立表面覆盖</div>
          <div><b>{{result.evaluation.joint_bad}}</b>身份或材料联合错误帧</div>
          <div><b>{{result.evaluation.arrival_p95_ms.toFixed(1)}} ms</b>视觉到达延迟 P95</div>
          <div><b>{{result.evaluation.motion.min_swept_envelope_clearance_m.toFixed(2)}} m</b>独立最小扫掠净空</div>
          <div><b>{{result.evaluation.motion.stale_applied.length}}</b>已应用的过期命令</div>
        </div>
        <p>独立评价 {{result.evaluation.passed ? '通过' : '未通过'}}。
          {{Object.entries(result.evaluation.checks).filter(([,ok]) => !ok).map(([name]) => name).join(' · ')}}</p>
        <video v-if="job?.state === 'completed'" ref="video" :key="selected" controls preload="metadata" :src="base+'/demo.mp4'" @timeupdate="seek"></video>
        <p v-if="currentFrame">实际帧 {{currentFrame.frame_index}} · {{currentFrame.actual_mode}} · 显示资格 {{currentFrame.status}}</p>
        <div class="photos">
          <article v-for="region in result.regions">
            <img v-if="region.photo" loading="lazy" :src="base+'/recording/'+region.photo.file" :alt="'实际拍摄区域 '+region.station">
            <strong>区域 {{region.station}} · {{region.complete ? '完成' : '未完成'}}</strong>
            <p>径向材料坐标 {{region.region_eta.map(n=>n.toFixed(3)).join('–')}}</p>
            <p v-if="region.reason" class="muted">{{region.reason}}</p>
          </article>
        </div>
        <details>
          <summary>策略结果对照</summary>
          <select v-model="comparedId" aria-label="对照实验">
            <option value="">选择另一份已完成的实验</option>
            <option v-for="other in jobs.filter(j=>j.state==='completed'&&j.id!==selected)" :value="other.id">{{other.id}}</option>
          </select>
          <table v-if="comparison">
            <thead><tr><th>场景 / 来源 / 策略</th><th>完成</th><th>覆盖</th><th>视觉延迟 P95</th></tr></thead>
            <tbody><tr v-for="item in [result,comparison]">
              <td>{{item.evaluation.scenario}} · {{item.source}} · {{arms[item.evaluation.arm]}}</td>
              <td>{{item.evaluation.region_count}}/{{item.regions.length}}</td>
              <td>{{percent(item.evaluation.coverage)}}</td><td>{{item.evaluation.arrival_p95_ms.toFixed(1)}} ms</td>
            </tr></tbody>
          </table>
          <p class="muted">请核对两次配置的曲线、场景和时长后再解释策略差异。</p>
        </details>
        <h3>下载证据</h3>
        <div class="controls links">
          <a v-for="[path,label] in [
            ['config.json','配置快照'],['result.json','独立结果'],['trajectory.csv','实际轨迹与角速度'],
            ['SUMMARY.json','独立安全与时效'],['surface_ledger.npz','覆盖账本'],['action_alignment.json','应用回执'],
            ['recording/observations.json','视觉估计及拒绝'],['recording/display_plans.json','在线参考与预测'],
            ['demo.frames.json','逐帧显示语义'],['demo.mp4','高清 Demo']].filter(([path])=>!path.startsWith('demo.') || job?.state==='completed')"
            :href="base+'/'+path" download>{{label}}</a>
        </div>
      </section>
    </main>`,
}).mount("#app");
