# تشغيل TinyGPT على Kaggle

## الحقيقة المهمة

الكود الحالي مكتوب بـNumPy، لذلك يعمل على CPU حتى لو اخترت TPU أو GPU. لا تختاري TPU وتتوقعي تسريعاً. للاستفادة من TPU نحتاج نقل النموذج إلى JAX/PyTorch، وهذا مشروع منفصل.

## الطريقة الأسهل

1. اضغطي **New Notebook** في Kaggle.
2. ارفعي المشروع كاملاً كملف ZIP، أو أضيفي مستودع GitHub إذا كان متاحاً.
3. فعّلي **Internet** من إعدادات Notebook.
4. افتحي Terminal داخل Kaggle أو شغّلي خلية:

```python
!pip install -q numpy pyarrow
!python /kaggle/working/codespaces-react/scripts/kaggle_train.py \
    --repo /kaggle/working/codespaces-react --hours 8
```

الناتج يحفظ في:

```text
/kaggle/working/chat_model/
```

وتابعي التدريب من:

```python
!cat /kaggle/working/chat_model/status.txt
```

## نقل الملفات

ارفعي هذه الملفات مع المشروع:

- `engine/`
- `data/dailydialog.parquet`
- `data/extra/soda_0.parquet` إن أردت SODA
- `models/chat_ckpt/` إذا أردت الاستئناف من checkpoint سابق
- `scripts/kaggle_train.py`

في نهاية الجلسة نزّلي مجلد `chat_model` وارفعيه إلى مشروعك المحلي.

## استخدام TPU أو GPU فعلياً

لا يمكن تشغيل NumPy الحالي على TPU مباشرة. يلزم إعادة كتابة `TinyGPT.forward` و`backward` و`AdamW` باستخدام JAX أو PyTorch، ثم تحويل البيانات إلى batches على accelerator. لذلك أوصي أولاً بتجربة الملف الحالي على Kaggle CPU، ثم تنفيذ نسخة JAX إذا كانت السرعة غير كافية.

لا ترفعي مفاتيح API أو بيانات خاصة إلى Kaggle. تحققي أيضاً من ترخيص كل dataset قبل التدريب.
