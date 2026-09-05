<script setup lang="ts">
import { onLaunch } from "@dcloudio/uni-app";
import { getVisitorId, ensureLogin } from "@/common/auth";

onLaunch(() => {
  // 生成访客标识；静默尝试微信登录（后端未配置 AppID 时自动降级，不打扰用户）
  getVisitorId();
  ensureLogin();
});
</script>
<style>
/* 全局暗色主题（与 Web 端星空暗色风一致） */
page {
  background-color: #0b1220;
  color: #e2e8f0;
  font-size: 28rpx;
  line-height: 1.6;
}
</style>
