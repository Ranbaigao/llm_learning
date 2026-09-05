<template>
  <view class="comments">
    <view class="section-title">评论（{{ total }}）</view>

    <!-- 发表框 -->
    <view class="form">
      <!-- 未填过昵称时：展示微信「头像昵称填写」能力 -->
      <view v-if="!nickname" class="profile-row">
        <button
          class="avatar-btn"
          open-type="chooseAvatar"
          @chooseavatar="onChooseAvatar"
        >
          <image v-if="avatar" class="avatar-img" :src="avatar" mode="aspectFill" />
          <text v-else>＋</text>
        </button>
        <input
          class="nickname-input"
          type="nickname"
          placeholder="点击获取/填写昵称"
          v-model="nickname"
          @blur="onNicknameBlur"
        />
      </view>
      <view v-else class="profile-row small">
        <image v-if="avatar" class="avatar-img" :src="avatar" mode="aspectFill" />
        <text class="who">{{ nickname }}</text>
        <text class="edit" @tap="nickname = ''">修改</text>
      </view>

      <textarea
        class="content-input"
        v-model="content"
        :placeholder="replyTo ? `回复 ${replyTo.nickname}：` : '写下你的评论…'"
        :maxlength="2000"
        auto-height
      />
      <view class="form-actions">
        <text v-if="replyTo" class="cancel-reply" @tap="replyTo = null"
          >取消回复</text
        >
        <button class="submit-btn" :disabled="submitting" @tap="submit">
          {{ submitting ? "提交中…" : "发表" }}
        </button>
      </view>
    </view>

    <!-- 评论列表 -->
    <view v-if="listLoading" class="hint">加载中…</view>
    <view v-else-if="flatComments.length === 0" class="hint"
      >还没有评论，来抢沙发～</view
    >
    <view v-else>
      <view
        v-for="item in flatComments"
        :key="item.comment.id"
        class="comment"
        :class="{ child: item.depth > 0 }"
      >
        <view class="comment-head">
          <text class="nickname">{{ item.comment.nickname }}</text>
          <text class="time">{{ formatTime(item.comment.created_at) }}</text>
        </view>
        <view class="comment-body">
          <text v-if="item.replyToName" class="reply-to"
            >回复 @{{ item.replyToName }}：</text
          >{{ item.comment.content }}
        </view>
        <view class="comment-actions">
          <text class="reply-btn" @tap="startReply(item.comment)">回复</text>
        </view>
      </view>
    </view>
  </view>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { request } from "@/common/request";
import {
  getVisitorId,
  getUser,
  saveProfile,
  ensureLogin,
  syncProfileToServer,
  isAuthFallback,
} from "@/common/auth";

interface CommentNode {
  id: number;
  article_id: number;
  nickname: string;
  content: string;
  parent_id: number | null;
  created_at: string;
  children: CommentNode[];
}

const props = defineProps<{ articleId: number }>();
const emit = defineEmits<{ (e: "commented", count: number): void }>();

const comments = ref<CommentNode[]>([]);
const listLoading = ref(false);
const submitting = ref(false);
const content = ref("");
const nickname = ref(getUser()?.nickname || "");
const avatar = ref(getUser()?.avatar || "");
const replyTo = ref<CommentNode | null>(null);

interface FlatComment {
  comment: CommentNode;
  depth: number; // 0/1，楼中楼只缩进一层
  replyToName: string;
}

/** 树 → 扁平列表；超过一层的回复不再加深缩进，用 @昵称 表示 */
const flatComments = computed<FlatComment[]>(() => {
  const out: FlatComment[] = [];
  const walk = (
    nodes: CommentNode[],
    depth: number,
    parentName: string
  ) => {
    for (const c of nodes) {
      out.push({ comment: c, depth, replyToName: parentName });
      walk(c.children || [], Math.min(depth + 1, 1), c.nickname);
    }
  };
  walk(comments.value, 0, "");
  return out;
});

const total = computed(() => flatComments.value.length);

function formatTime(s: string): string {
  return s ? s.replace("T", " ").slice(0, 16) : "";
}

async function loadComments() {
  if (!props.articleId) return;
  listLoading.value = true;
  try {
    comments.value = await request<CommentNode[]>({
      url: `/articles/${props.articleId}/comments`,
    });
  } catch (e) {
    // request 已 toast
  } finally {
    listLoading.value = false;
  }
}

watch(() => props.articleId, loadComments, { immediate: true });

function onChooseAvatar(e: any) {
  // 头像为临时文件路径，仅本地展示；后端无上传接口，不落库
  avatar.value = e.detail.avatarUrl;
  if (nickname.value) saveProfile(nickname.value, avatar.value);
}

function onNicknameBlur() {
  if (nickname.value.trim()) {
    saveProfile(nickname.value.trim(), avatar.value || undefined);
    syncProfileToServer(nickname.value.trim());
  }
}

function startReply(c: CommentNode) {
  replyTo.value = c;
}

async function submit() {
  const text = content.value.trim();
  const name = nickname.value.trim();
  if (!name) {
    uni.showToast({ title: "请先填写昵称", icon: "none" });
    return;
  }
  if (!text) {
    uni.showToast({ title: "评论内容不能为空", icon: "none" });
    return;
  }
  // 首次发表评论时触发微信登录；503（未配 AppID）时静默降级为手动昵称模式
  submitting.value = true;
  try {
    await ensureLogin();
    saveProfile(name, avatar.value || undefined);
    await request<CommentNode>({
      url: `/articles/${props.articleId}/comments`,
      method: "POST",
      data: {
        visitor_id: getVisitorId(),
        nickname: name,
        content: text,
        parent_id: replyTo.value ? replyTo.value.id : undefined,
      },
    });
    content.value = "";
    replyTo.value = null;
    uni.showToast({ title: "发表成功", icon: "success" });
    await loadComments();
    emit("commented", total.value);
  } catch (e) {
    // 422 校验 / 429 限流等错误 request 已 toast detail
  } finally {
    submitting.value = false;
  }
}

// 供父组件了解当前登录模式（未强制要求使用）
defineExpose({ isAuthFallback });
</script>

<style>
.comments {
  padding: 24rpx;
  border-top: 1rpx solid #1e2a4a;
  margin-top: 32rpx;
}
.section-title {
  font-size: 30rpx;
  font-weight: 600;
  color: #f1f5f9;
  margin-bottom: 20rpx;
}
.form {
  background: #111c33;
  border: 1rpx solid #1e2a4a;
  border-radius: 16rpx;
  padding: 20rpx;
  margin-bottom: 24rpx;
}
.profile-row {
  display: flex;
  align-items: center;
  gap: 16rpx;
  margin-bottom: 16rpx;
}
.profile-row.small {
  font-size: 24rpx;
  color: #94a3b8;
}
.avatar-btn {
  width: 64rpx;
  height: 64rpx;
  border-radius: 50%;
  padding: 0;
  margin: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #1e2a4a;
  color: #38bdf8;
  font-size: 36rpx;
  line-height: 1;
}
.avatar-btn::after {
  border: none;
}
.avatar-img {
  width: 64rpx;
  height: 64rpx;
  border-radius: 50%;
}
.nickname-input {
  flex: 1;
  background: #0b1220;
  border: 1rpx solid #1e2a4a;
  border-radius: 12rpx;
  padding: 12rpx 20rpx;
  color: #e2e8f0;
  font-size: 26rpx;
}
.who {
  color: #e2e8f0;
}
.edit {
  color: #38bdf8;
}
.content-input {
  width: 100%;
  min-height: 120rpx;
  background: #0b1220;
  border: 1rpx solid #1e2a4a;
  border-radius: 12rpx;
  padding: 16rpx 20rpx;
  color: #e2e8f0;
  font-size: 26rpx;
  box-sizing: border-box;
}
.form-actions {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  gap: 24rpx;
  margin-top: 16rpx;
}
.cancel-reply {
  color: #94a3b8;
  font-size: 24rpx;
}
.submit-btn {
  background: #38bdf8;
  color: #0b1220;
  font-size: 26rpx;
  font-weight: 600;
  border-radius: 999rpx;
  padding: 0 40rpx;
  margin: 0;
  line-height: 64rpx;
}
.submit-btn[disabled] {
  opacity: 0.5;
}
.comment {
  padding: 20rpx 0;
  border-bottom: 1rpx solid #1e2a4a;
}
.comment.child {
  margin-left: 48rpx;
  border-left: 4rpx solid #1e2a4a;
  padding-left: 20rpx;
}
.comment-head {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8rpx;
}
.nickname {
  color: #38bdf8;
  font-size: 26rpx;
  font-weight: 600;
}
.time {
  color: #64748b;
  font-size: 22rpx;
}
.comment-body {
  color: #e2e8f0;
  font-size: 27rpx;
  word-break: break-word;
}
.reply-to {
  color: #94a3b8;
}
.comment-actions {
  margin-top: 8rpx;
}
.reply-btn {
  color: #38bdf8;
  font-size: 24rpx;
}
.hint {
  text-align: center;
  color: #94a3b8;
  padding: 40rpx 0;
}
</style>
