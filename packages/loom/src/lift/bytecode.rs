//! A function's bytecode to the structurer's instructions, and each expression in it to loom's
//! syntax.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
use serde_json::Value;

use super::code::{Code, Insn};
use super::input::{field_name, prop_ref, str_of, Prop, PropTy, Ref};
use super::program::{invented, own_name, short, Blueprint, Callee, Dir, Program};
use crate::builtins;
use crate::syntax::{BinaryOp, Expr, ExprKind, Span};

/// What one bytecode statement does, before the structurer puts it in place.
#[derive(Debug, Clone, PartialEq)]
pub enum Lx {
    /// `target = value`. `pure` when the value only computes, so reading it again later gives
    /// the same.
    Assign { target: Expr, value: Expr, pure: bool },
    /// `let name = call`, for a call with several outputs, which are read as its members.
    Record { name: String, call: Expr },
    /// A call on its own.
    Do(Expr),
    /// A delegate variable set to one of an object's functions, for a later `AddDelegate`.
    Bind { var: String, object: Expr, function: String },
    /// A delegate variable added to a dispatcher.
    AddDelegate { dispatcher: Expr, var: String },
}

pub type Stmts = Vec<Lx>;

/// One function's bytecode being read: the Blueprint it belongs to, and its locals.
pub struct Reader<'p, 'a> {
    pub program: &'p Program<'a>,
    pub bp: &'p Blueprint,
    /// The function's parameters and locals, by cooked name.
    pub locals: HashMap<String, Prop>,
    /// Names given to locals, by cooked name: a parameter's, or an event's parameter for the
    /// event graph's copy of it.
    pub names: HashMap<String, String>,
    /// Locals set by a call's out argument, which must be declared as variables.
    pub out_args: HashSet<String>,
    /// A function library's function has no self.
    pub library: bool,
    /// Every name the function's code may use, so a name the lifter makes up takes none of them.
    taken: HashSet<String>,
}

fn e(kind: ExprKind) -> Expr {
    Expr { kind, span: Span::default() }
}

fn name(n: &str) -> Expr {
    e(ExprKind::Name(n.to_string()))
}

impl<'p, 'a> Reader<'p, 'a> {
    pub fn new(program: &'p Program<'a>, bp: &'p Blueprint, function: &Value) -> Result<Self> {
        let mut locals = HashMap::new();
        for p in function.get("ChildProperties").and_then(Value::as_array).into_iter().flatten() {
            let prop = Prop::of(p)?;
            locals.insert(prop.name.clone(), prop);
        }
        let library = program.is_subclass(&bp.class, "/Script/Engine.BlueprintFunctionLibrary");
        // The Blueprint's own names, which a local must not take.
        let mut taken: HashSet<String> = bp.functions.values().map(|f| f.name.clone()).collect();
        if let Some(class) = bp.package.exports.iter().find(|e| matches!(str_of(e, "Type"), Some("BlueprintGeneratedClass" | "WidgetBlueprintGeneratedClass"))) {
            for p in class.get("ChildProperties").and_then(Value::as_array).into_iter().flatten() {
                if let Some(n) = str_of(p, "Name") {
                    taken.insert(n.to_string());
                }
            }
        }
        let mut cooked: Vec<&String> = locals.keys().collect();
        cooked.sort();
        let mut names = HashMap::new();
        // Parameters keep their names; the compiler's locals are named for what they hold.
        for c in cooked.iter().filter(|c| locals[**c].has("Parm")) {
            let n = own_name(c)?;
            taken.insert(n.clone());
            names.insert((*c).clone(), n);
        }
        for c in cooked.iter().filter(|c| !locals[**c].has("Parm")) {
            let n = fresh(&mut taken, &readable(c));
            names.insert((*c).clone(), n);
        }
        // A user's local keeps its name, which must be one a source can write.
        for n in names.values() {
            own_name(n)?;
        }
        Ok(Reader { program, bp, locals, names, out_args: HashSet::new(), library, taken })
    }

    /// The function's bytecode as instructions.
    pub fn code(&mut self, function: &Value) -> Result<Code<Stmts, Expr>> {
        let mut insns = Vec::new();
        for s in function.get("ScriptBytecode").and_then(Value::as_array).into_iter().flatten() {
            let at = s.get("StatementIndex").and_then(Value::as_u64).ok_or_else(|| refusal!(Unexpected, "a statement with no offset"))? as usize;
            let insn = self.statement(s).with_context(|| format!("at offset {at}"))?;
            insns.push((at, insn));
        }
        Ok(Code { insns })
    }

    /// The name a local is written as.
    pub fn local_name(&self, cooked: &str) -> String {
        self.names.get(cooked).cloned().unwrap_or_else(|| cooked.to_string())
    }

    fn statement(&mut self, s: &Value) -> Result<Insn<Stmts, Expr>> {
        let token = str_of(s, "Token").unwrap_or_default();
        let offset = |key: &str| -> Result<usize> {
            s.get(key).and_then(Value::as_u64).map(|v| v as usize).ok_or_else(|| refusal!(Unexpected, "{token} with no {key}"))
        };
        Ok(match token {
            "EX_Jump" => Insn::Jump(offset("CodeOffset")?),
            "EX_JumpIfNot" => Insn::JumpIfNot(self.expr(field(s, "BooleanExpression")?)?, offset("CodeOffset")?),
            "EX_PushExecutionFlow" => Insn::Push(offset("PushingAddress")?),
            "EX_PopExecutionFlow" => Insn::Pop,
            "EX_PopExecutionFlowIfNot" => Insn::PopIfNot(self.expr(field(s, "BooleanExpression")?)?),
            "EX_ComputedJump" => Insn::ComputedJump,
            "EX_Return" => Insn::Return,
            "EX_EndOfScript" => Insn::End,
            // A breakpoint's or a wire's marker in a development cook.
            "EX_Tracepoint" | "EX_WireTracepoint" | "EX_InstrumentationEvent" | "EX_Nothing" => Insn::Stmt(Vec::new()),
            _ => {
                let mut out = Vec::new();
                if let Some(resume) = self.effect(s, &mut out)? {
                    Insn::Latent { call: out, resume }
                } else {
                    Insn::Stmt(out)
                }
            }
        })
    }

    /// A statement that does something. Returns the offset a latent call resumes at.
    fn effect(&mut self, s: &Value, out: &mut Stmts) -> Result<Option<usize>> {
        let token = str_of(s, "Token").unwrap_or_default();
        match token {
            "EX_Let" | "EX_LetBool" | "EX_LetObj" | "EX_LetWeakObjPtr" | "EX_LetDelegate" | "EX_LetMulticastDelegate" => {
                let target = self.place(field(s, "Variable")?)?;
                let value = field(s, "Expression")?;
                self.assign(target, value, out)?;
            }
            "EX_LetValueOnPersistentFrame" => {
                let (_, dest) = prop_ref(field(s, "DestinationProperty")?)?;
                let target = name(&self.local_name(&dest));
                self.assign(target, field(s, "AssignmentExpression")?, out)?;
            }
            "EX_SetArray" | "EX_SetSet" => {
                let key = if token == "EX_SetArray" { "AssigningProperty" } else { "SetProperty" };
                let target = self.place(field(s, key)?)?;
                if token == "EX_SetArray" && holds_map_const(s) {
                    let items: Vec<Value> = s.get("Elements").and_then(Value::as_array).cloned().unwrap_or_default();
                    return self.fill_array(target, &items, out).map(|_| None);
                }
                let mut items = Vec::new();
                for item in s.get("Elements").and_then(Value::as_array).into_iter().flatten() {
                    items.push(self.expr(item)?);
                }
                let value = if items.is_empty() { e(ExprKind::EmptyArray) } else { e(ExprKind::ArrayLiteral(items)) };
                out.push(Lx::Assign { target, value, pure: false });
            }
            "EX_SetMap" => {
                // A map literal is only a variable's default in loom, so the map is emptied and
                // filled an entry at a time.
                let target = self.place(field(s, "MapProperty")?)?;
                out.push(Lx::Do(e(ExprKind::Method(Box::new(target.clone()), "Clear".into(), Vec::new()))));
                for entry in s.get("Values").and_then(Value::as_array).into_iter().flatten() {
                    let key = self.expr(field(entry, "Key")?)?;
                    let value = self.expr(field(entry, "Value")?)?;
                    out.push(Lx::Assign { target: e(ExprKind::Index(Box::new(target.clone()), Box::new(key))), value, pure: false });
                }
            }
            "EX_BindDelegate" => {
                let var = self.local_var(field(s, "Delegate")?)?;
                let object = self.expr(field(s, "ObjectTerm")?)?;
                let function = str_of(s, "FunctionName").ok_or_else(|| refusal!(Unexpected, "a delegate bound to no function"))?;
                let function = match self.program.function(&self.bp.class, function) {
                    Some((_, Callee::Lifted(f))) => f.name,
                    _ => function.to_string(),
                };
                out.push(Lx::Bind { var, object, function });
            }
            "EX_AddMulticastDelegate" => {
                let dispatcher = self.expr(field(s, "MulticastDelegate")?)?;
                let var = self.local_var(field(s, "Delegate")?)?;
                out.push(Lx::AddDelegate { dispatcher, var });
            }
            _ => return self.call_statement(s, None, out),
        }
        Ok(None)
    }

    /// `target = value`, where the value may be a call whose outputs go to other variables too.
    fn assign(&mut self, target: Expr, value: &Value, out: &mut Stmts) -> Result<()> {
        if holds_map_const(value) {
            return self.fill(target, value, out);
        }
        if is_call(value) {
            let call = self.call(value, None)?;
            return self.finish_call(call, Some(target), out).map(|_| ());
        }
        let v = self.expr(value)?;
        let pure = self.is_pure(value);
        out.push(Lx::Assign { target, value: v, pure });
        Ok(())
    }

    /// Sets a variable to a constant that holds a map, statement by statement, since loom writes
    /// a map literal only as a variable's default.
    fn fill(&mut self, target: Expr, value: &Value, out: &mut Stmts) -> Result<()> {
        match str_of(value, "Token") {
            Some("EX_MapConst") => {
                out.push(Lx::Do(e(ExprKind::Method(Box::new(target.clone()), "Clear".into(), Vec::new()))));
                for entry in value.get("Values").and_then(Value::as_array).into_iter().flatten() {
                    let key = self.expr(field(entry, "Key")?)?;
                    let v = field(entry, "Value")?;
                    self.fill(e(ExprKind::Index(Box::new(target.clone()), Box::new(key))), v, out)?;
                }
                Ok(())
            }
            Some("EX_StructConst") if holds_map_const(value) => {
                let path = Ref::of(field(value, "Struct")?)?.path();
                let fields = self.program.struct_fields(&path)?;
                let values: Vec<&Value> = value.get("Properties").and_then(Value::as_array).map(|a| a.iter().collect()).unwrap_or_default();
                if fields.len() != values.len() {
                    unexpected!("a {} constant with {} values for {} fields", short(&path), values.len(), fields.len());
                }
                // The struct's plain fields at once, then each field that holds a map.
                let mut made = Vec::new();
                let mut later = Vec::new();
                for ((n, ty), v) in fields.iter().zip(values) {
                    if holds_map_const(v) {
                        later.push((n.clone(), v));
                    } else {
                        made.push((n.clone(), self.arg(v, Some(ty))?));
                    }
                }
                out.push(Lx::Assign { target: target.clone(), value: e(ExprKind::Make(self.program.struct_name(&path)?, made)), pure: false });
                for (n, v) in later {
                    self.fill(e(ExprKind::Member(Box::new(target.clone()), n)), v, out)?;
                }
                Ok(())
            }
            Some("EX_ArrayConst") if holds_map_const(value) => {
                let items: Vec<Value> = value.get("Values").or_else(|| value.get("Elements")).and_then(Value::as_array).cloned().unwrap_or_default();
                self.fill_array(target, &items, out)
            }
            _ => {
                let v = self.expr(value)?;
                out.push(Lx::Assign { target, value: v, pure: false });
                Ok(())
            }
        }
    }

    /// An array sized to its items, then each item set in place.
    fn fill_array(&mut self, target: Expr, items: &[Value], out: &mut Stmts) -> Result<()> {
        out.push(Lx::Assign { target: target.clone(), value: e(ExprKind::EmptyArray), pure: false });
        let size = e(ExprKind::Int(items.len() as i64));
        out.push(Lx::Do(e(ExprKind::Method(Box::new(target.clone()), "Resize".into(), vec![size]))));
        for (i, item) in items.iter().enumerate() {
            let at = e(ExprKind::Index(Box::new(target.clone()), Box::new(e(ExprKind::Int(i as i64)))));
            self.fill(at, item, out)?;
        }
        Ok(())
    }

    fn call_statement(&mut self, s: &Value, target: Option<Expr>, out: &mut Stmts) -> Result<Option<usize>> {
        if !is_call(s) {
            not_yet!("{} as a statement", str_of(s, "Token").unwrap_or("?"));
        }
        let call = self.call(s, None)?;
        self.finish_call(call, target, out)
    }

    /// Puts a call in place: on its own, or assigned, with its outputs read from a record.
    fn finish_call(&mut self, call: Call, target: Option<Expr>, out: &mut Stmts) -> Result<Option<usize>> {
        let Call { expr, outputs, value_to, reads, returns, pure, resume } = call;
        let target = match (target, value_to) {
            (Some(_), Some(_)) => unexpected!("a call whose value goes to two variables"),
            (t, v) => t.or(v),
        };
        if outputs.is_empty() {
            match target {
                Some(t) => out.push(Lx::Assign { target: t, value: expr, pure }),
                // A call that only computes does nothing on its own, and loom refuses one there.
                None if pure => {}
                None => out.push(Lx::Do(expr)),
            }
        } else {
            // A call's outputs are named for the call, as its return value is.
            let record = fresh(&mut self.taken, &lower_first(&invented(&call_name(&expr))));
            out.push(Lx::Record { name: record.clone(), call: expr });
            let member = |m: &str| e(ExprKind::Member(Box::new(name(&record)), m.to_string()));
            if let Some(t) = target {
                if !returns {
                    unexpected!("a call with no return value assigned to a variable");
                }
                out.push(Lx::Assign { target: t, value: member("ReturnValue"), pure: true });
            }
            for (target, output) in outputs {
                out.push(Lx::Assign { target, value: member(&output), pure: true });
            }
        }
        for (target, value) in reads {
            out.push(Lx::Assign { target, value, pure: true });
        }
        Ok(resume)
    }

    /// A variable an assignment or an out argument writes.
    /// Where a value is put. A field the project's types make read-only has no pin on any node
    /// that could set it, though the project the mod was built in let it be set.
    fn place(&mut self, v: &Value) -> Result<Expr> {
        if str_of(v, "Token") == Some("EX_StructMemberContext") {
            if let (Some(owner), cooked) = prop_ref(field(v, "Property")?)? {
                if owner.from_script() {
                    let path = owner.path();
                    let spelled = self.program.field_spelling(&path, &cooked);
                    if self.program.db.field_read_only(&path, &spelled) {
                        missing!("{}'s {spelled} set, which the project's types make read-only", short(&path));
                    }
                }
            }
        }
        self.expr(v)
    }

    fn local_var(&mut self, v: &Value) -> Result<String> {
        match str_of(v, "Token") {
            Some("EX_LocalVariable" | "EX_LocalOutVariable") => Ok(prop_ref(field(v, "Variable")?)?.1),
            other => not_yet!("a delegate held in {other:?}"),
        }
    }

    /// Whether an expression only computes a value.
    pub fn is_pure(&self, v: &Value) -> bool {
        match str_of(v, "Token").unwrap_or_default() {
            "EX_CallMath" => true,
            "EX_FinalFunction" | "EX_LocalFinalFunction" | "EX_VirtualFunction" | "EX_LocalVirtualFunction" => {
                self.callee(v, None).map(|(_, c)| match c {
                    Callee::Lifted(f) => f.pure,
                    Callee::Native(sig) => sig.pure,
                }).unwrap_or(false)
            }
            "EX_Context" | "EX_Context_FailSilent" | "EX_ClassContext" => {
                let ctx = v.get("ContextExpression").unwrap_or(&Value::Null);
                if is_call(ctx) {
                    let class = self.class_of(v.get("ObjectExpression").unwrap_or(&Value::Null));
                    self.callee(ctx, class.as_deref()).map(|(_, c)| match c {
                        Callee::Lifted(f) => f.pure,
                        Callee::Native(sig) => sig.pure,
                    }).unwrap_or(false)
                } else {
                    true
                }
            }
            "EX_DynamicCast" | "EX_MetaCast" => false,
            _ => true,
        }
    }

    /// The function a call reaches, and the class it is looked up on.
    fn callee(&self, v: &Value, on: Option<&str>) -> Result<(String, Callee)> {
        let token = str_of(v, "Token").unwrap_or_default();
        let (class, function) = match token {
            "EX_CallMath" | "EX_FinalFunction" | "EX_LocalFinalFunction" => Ref::of(field(v, "Function")?)?.function()?,
            "EX_VirtualFunction" | "EX_LocalVirtualFunction" => {
                let function = match v.get("Function") {
                    Some(Value::String(s)) => s.clone(),
                    _ => str_of(v, "VirtualFunctionName").ok_or_else(|| refusal!(Unexpected, "a virtual call with no function"))?.to_string(),
                };
                let class = match on {
                    Some(c) => c.to_string(),
                    None => self.bp.class.clone(),
                };
                (class, function)
            }
            other => unexpected!("{other} where a call should be"),
        };
        // A final call names the class that declares the function; a call through an object
        // looks it up on the object's class.
        let lookup = if token.contains("Final") || token == "EX_CallMath" { class.clone() } else { class };
        if let Some(found) = self.program.function(&lookup, &function) {
            return Ok(found);
        }
        // A class the project's types lack says so, rather than that its function is unknown.
        self.program.class_name(&lookup)?;
        missing!("{}.{function}, which the project's types do not have", short(&lookup))
    }

    /// The class of the object an expression gives, where it can be told.
    fn class_of(&self, v: &Value) -> Option<String> {
        match str_of(v, "Token")? {
            "EX_Self" => Some(self.bp.class.clone()),
            "EX_LocalVariable" | "EX_LocalOutVariable" | "EX_InstanceVariable" | "EX_DefaultVariable" => {
                let prop = match v.pointer("/Variable/Property") {
                    Some(p) => PropTy::of(p).ok()?,
                    None => {
                        let (owner, n) = prop_ref(v.get("Variable")?).ok()?;
                        PropTy::from_ty(&self.program.db.property(&owner?.path(), &n)?.ty).ok()?
                    }
                };
                match prop {
                    PropTy::Object(c) | PropTy::Interface(c) => Some(c),
                    _ => None,
                }
            }
            "EX_ObjectConst" => {
                let r = Ref::of(v.get("Value")?).ok()?;
                // A class default object, Default__X, stands for its class.
                let class = r.name.strip_prefix("Default__")?;
                Some(format!("{}.{class}", r.package))
            }
            "EX_Context" | "EX_Context_FailSilent" => self.class_of(v.get("ContextExpression")?),
            "EX_DynamicCast" => Some(Ref::of(v.get("ClassPtr").or_else(|| v.get("InterfaceClass"))?).ok()?.path()),
            "EX_InterfaceContext" => self.class_of(v.get("InterfaceValue")?),
            _ => None,
        }
    }

    /// A call: its expression, and the variables that receive its out arguments. `object` is
    /// the expression the call is made through, when it is.
    fn call(&mut self, v: &Value, object: Option<&Value>) -> Result<Call> {
        let token = str_of(v, "Token").unwrap_or_default();
        if matches!(token, "EX_Context" | "EX_Context_FailSilent" | "EX_ClassContext") {
            return self.call(field(v, "ContextExpression")?, Some(field(v, "ObjectExpression")?));
        }
        let on = object.and_then(|o| self.class_of(o));
        let (owner, callee) = self.callee(v, on.as_deref())?;
        let is_static = match &callee {
            Callee::Lifted(f) => f.is_static,
            Callee::Native(sig) => sig.is_static,
        };
        // A static function is called on its class, whatever object the bytecode names, which is
        // usually the class's default object.
        let object = match object {
            Some(o) if !is_static => Some(self.expr(o)?),
            _ => None,
        };
        let args: Vec<&Value> = v.get("Parameters").and_then(Value::as_array).map(|a| a.iter().collect()).unwrap_or_default();
        match callee {
            Callee::Lifted(f) => self.lifted_call(&owner, &f, &args, object),
            Callee::Native(sig) => self.native_call(&owner, &sig, v, &args, object),
        }
    }

    fn lifted_call(&mut self, owner: &str, f: &super::program::Function, args: &[&Value], object: Option<Expr>) -> Result<Call> {
        // A return parameter is the call's value, not an argument.
        let params: Vec<_> = f.params.iter().filter(|p| p.cooked != "__WorldContext" && !p.ret).collect();
        let passed: Vec<&Value> = if f.params.iter().any(|p| p.cooked == "__WorldContext") {
            // A library function's compiled calls pass the world context first.
            args.iter().skip(1).copied().collect()
        } else {
            args.to_vec()
        };
        if params.len() != passed.len() {
            unexpected!("a call of {} with {} arguments, which takes {}", f.name, passed.len(), params.len());
        }
        let single_return = returns_value(f);
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut returns = f.params.iter().any(|p| p.ret);
        let mut ret_target = None;
        for (p, a) in params.iter().zip(passed) {
            match p.dir {
                Dir::In | Dir::Ref => inputs.push(self.arg(a, Some(&p.ty))?),
                Dir::Out => {
                    let target = self.out_arg(a)?;
                    if single_return {
                        ret_target = Some(target);
                        returns = true;
                    } else {
                        outputs.push((target, p.name.clone()));
                    }
                }
            }
        }
        let expr = self.call_expr(owner, &f.name, f.is_static, inputs, object)?;
        // The only output is written as the function's return value, so the call is assigned.
        Ok(Call { outputs, value_to: ret_target, ..Call::value(expr, returns, f.pure) })
    }

    fn native_call(&mut self, owner: &str, sig: &crate::types::CompiledSignature, v: &Value, args: &[&Value], object: Option<Expr>) -> Result<Call> {
        let function = call_function_name(v)?;
        if sig.params.len() != args.len() {
            unexpected!("a call of {}.{function} with {} arguments, which takes {}", short(owner), args.len(), sig.params.len());
        }
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut resume = None;
        // A byte beside an enum's value is that enum's, as a Switch on Enum compares its
        // selection with each case's byte.
        let beside = args.iter().find_map(|a| enum_of(a));
        for (p, a) in sig.params.iter().zip(args) {
            if p.hidden {
                if sig.latent && p.ty == "struct:/Script/Engine.LatentActionInfo" {
                    resume = Some(latent_resume(a)?);
                }
                continue;
            }
            match p.dir.as_str() {
                "out" => outputs.push((self.out_arg(a)?, p.name.clone())),
                _ => {
                    let ty = match (PropTy::from_dump(&p.ty).ok(), &beside) {
                        (Some(PropTy::Byte), Some(path)) => Some(PropTy::Enum(path.clone())),
                        (ty, _) => ty,
                    };
                    inputs.push(self.arg(a, ty.as_ref())?);
                }
            }
        }
        if let Some(call) = self.builtin(owner, &function, &inputs, &outputs, sig.has_return, sig.pure)? {
            return Ok(call);
        }
        let expr = self.call_expr(owner, &function, sig.is_static, inputs, object)?;
        Ok(Call { outputs, resume, ..Call::value(expr, sig.has_return, sig.pure) })
    }

    /// A call of one of the engine's functions that loom writes as an operator, a conversion or
    /// a method, as the reference's page of built-ins lists them.
    fn builtin(&mut self, owner: &str, function: &str, inputs: &[Expr], outputs: &[(Expr, String)], returns: bool, pure: bool) -> Result<Option<Call>> {
        let pure_value = |expr| Some(Call::value(expr, true, true));
        for op in builtins::OPERATORS {
            if let builtins::Lowering::Library { class, function: f, .. } = op.lowering {
                if class == owner && f == function && inputs.len() == 2 {
                    return Ok(pure_value(e(ExprKind::Binary(op.op, Box::new(inputs[0].clone()), Box::new(inputs[1].clone())))));
                }
            }
        }
        for u in builtins::UNARY {
            if owner == builtins::MATH && u.function == function && !u.zero_first && inputs.len() == 1 {
                return Ok(pure_value(e(ExprKind::Unary(u.op, Box::new(inputs[0].clone())))));
            }
        }
        for c in builtins::CONVERSIONS {
            if let [step] = c.steps {
                if step.class == owner && step.function == function && inputs.len() == 1 {
                    return Ok(pure_value(e(ExprKind::Call(c.to.to_string(), vec![inputs[0].clone()], false))));
                }
            }
        }
        let prefix = match owner {
            builtins::ARRAY => "Array_",
            builtins::SET => "Set_",
            builtins::MAP => "Map_",
            _ => return Ok(None),
        };
        match function {
            // Reading an item is indexing.
            "Array_Get" if inputs.len() == 2 && outputs.len() == 1 => {
                let item = e(ExprKind::Index(Box::new(inputs[0].clone()), Box::new(inputs[1].clone())));
                return Ok(Some(Call { value_to: Some(outputs[0].0.clone()), ..Call::value(item, true, true) }));
            }
            "Map_Find" if inputs.len() == 2 && outputs.len() == 1 => {
                // What the map holds for the key, or the value type's zero, which is what
                // map[key] gives; and whether it holds one.
                let value = e(ExprKind::Index(Box::new(inputs[0].clone()), Box::new(inputs[1].clone())));
                let found = e(ExprKind::Method(Box::new(inputs[0].clone()), "Contains".into(), vec![inputs[1].clone()]));
                return Ok(Some(Call { reads: vec![(outputs[0].0.clone(), value)], ..Call::value(found, true, true) }));
            }
            _ => {}
        }
        // Any other function of the library whose name has its prefix is a method of the
        // container it is called on, which is its first argument: Array_RemoveItem(Items, 4) is
        // Items.RemoveItem(4). One without the prefix, such as SortIntArray, is called on the
        // library.
        let Some(method) = function.strip_prefix(prefix) else { return Ok(None) };
        let (target, rest) = inputs.split_first().ok_or_else(|| refusal!(Unexpected, "{function} with no container"))?;
        let expr = e(ExprKind::Method(Box::new(target.clone()), method.to_string(), rest.to_vec()));
        Ok(Some(Call { outputs: outputs.to_vec(), ..Call::value(expr, returns, pure) }))
    }

    /// `Class.f(args)` for a static function, `object.f(args)` for one called on an object, and
    /// `f(args)` for one of this Blueprint's own.
    fn call_expr(&self, owner: &str, function: &str, is_static: bool, args: Vec<Expr>, object: Option<Expr>) -> Result<Expr> {
        if is_static {
            if self.library && owner == self.bp.class {
                return Ok(e(ExprKind::Call(function.to_string(), args, builtins::function(function).is_some())));
            }
            let class = self.program.class_name(owner)?;
            return Ok(e(ExprKind::Method(Box::new(name(&class)), function.to_string(), args)));
        }
        match object {
            Some(o) if !matches!(&o.kind, ExprKind::Name(n) if n == "self") => Ok(e(ExprKind::Method(Box::new(o), function.to_string(), args))),
            _ => Ok(e(ExprKind::Call(function.to_string(), args, builtins::function(function).is_some()))),
        }
    }

    /// An argument, written for the parameter's type where a literal needs it, such as a byte
    /// that is an enum's value.
    fn arg(&mut self, v: &Value, ty: Option<&PropTy>) -> Result<Expr> {
        if let (Some("EX_ByteConst" | "EX_IntConst"), Some(PropTy::Enum(path))) = (str_of(v, "Token"), ty) {
            let n = v.get("Value").and_then(Value::as_i64).unwrap_or(0);
            return self.program.enum_number(path, n);
        }
        self.expr(v)
    }

    /// A variable an out argument writes.
    fn out_arg(&mut self, v: &Value) -> Result<Expr> {
        if matches!(str_of(v, "Token"), Some("EX_LocalVariable" | "EX_LocalOutVariable")) {
            self.out_args.insert(prop_ref(field(v, "Variable")?)?.1);
        }
        self.expr(v)
    }

    pub fn expr(&mut self, v: &Value) -> Result<Expr> {
        let token = str_of(v, "Token").ok_or_else(|| refusal!(Unexpected, "an expression with no token: {v}"))?;
        let value = || v.get("Value").ok_or_else(|| refusal!(Unexpected, "{token} with no value"));
        Ok(match token {
            "EX_LocalVariable" | "EX_LocalOutVariable" => {
                let (_, n) = prop_ref(field(v, "Variable")?)?;
                if n == "__WorldContext" {
                    cannot!("the world context read other than as a call's");
                }
                name(&self.local_name(&n))
            }
            "EX_InstanceVariable" | "EX_DefaultVariable" => {
                let (_, n) = prop_ref(field(v, "Variable")?)?;
                name(&n)
            }
            "EX_Context" | "EX_Context_FailSilent" | "EX_ClassContext" => {
                let ctx = field(v, "ContextExpression")?;
                if is_call(ctx) {
                    let call = self.call(ctx, Some(field(v, "ObjectExpression")?))?;
                    if !call.outputs.is_empty() || call.value_to.is_some() || !call.reads.is_empty() {
                        unexpected!("a call with outputs inside an expression");
                    }
                    return Ok(call.expr);
                }
                let object = self.expr(field(v, "ObjectExpression")?)?;
                let member = match str_of(ctx, "Token") {
                    Some("EX_InstanceVariable" | "EX_DefaultVariable" | "EX_LocalVariable") => {
                        prop_ref(field(ctx, "Variable")?)?.1
                    }
                    other => not_yet!("a member read through {other:?}"),
                };
                e(ExprKind::Member(Box::new(object), member))
            }
            "EX_StructMemberContext" => {
                let (owner, cooked) = prop_ref(field(v, "Property")?)?;
                let member = match &owner {
                    Some(o) if !o.from_script() => field_name(&cooked).to_string(),
                    Some(o) => self.program.field_spelling(&o.path(), &cooked),
                    None => cooked.clone(),
                };
                e(ExprKind::Member(Box::new(self.expr(field(v, "StructExpression")?)?), member))
            }
            "EX_ArrayGetByRef" => e(ExprKind::Index(Box::new(self.expr(field(v, "ArrayVariable")?)?), Box::new(self.expr(field(v, "ArrayIndex")?)?))),
            "EX_Self" => {
                if self.library {
                    unexpected!("self in a function library");
                }
                name("self")
            }
            "EX_IntConst" | "EX_Int64Const" | "EX_UInt64Const" | "EX_ByteConst" | "EX_IntConstByte" => {
                e(ExprKind::Int(value()?.as_i64().ok_or_else(|| refusal!(Unexpected, "{token} that is not a whole number"))?))
            }
            "EX_IntZero" => e(ExprKind::Int(0)),
            "EX_IntOne" => e(ExprKind::Int(1)),
            "EX_FloatConst" | "EX_DoubleConst" => e(ExprKind::Float(value()?.as_f64().ok_or_else(|| refusal!(Unexpected, "{token} that is not a number"))?)),
            "EX_StringConst" | "EX_UnicodeStringConst" | "EX_NameConst" => e(ExprKind::Str(value()?.as_str().unwrap_or_default().to_string())),
            "EX_TextConst" => {
                let t = value()?;
                let s = str_of(t, "SourceString")
                    .or_else(|| t.pointer("/Text/SourceString").and_then(Value::as_str))
                    .or_else(|| t.pointer("/SourceString/Value").and_then(Value::as_str))
                    .or_else(|| t.pointer("/KeyString/Value").and_then(Value::as_str).and(Some("")))
                    .or_else(|| t.as_str())
                    .ok_or_else(|| refusal!(NotYet, "a text constant kept this way: {t}"))?;
                e(ExprKind::Str(s.to_string()))
            }
            "EX_True" => e(ExprKind::Bool(true)),
            "EX_False" => e(ExprKind::Bool(false)),
            "EX_BitFieldConst" => e(ExprKind::Bool(v.get("ConstValue").and_then(Value::as_i64).unwrap_or(0) != 0)),
            "EX_NoObject" | "EX_NoInterface" => e(ExprKind::Null),
            "EX_ObjectConst" => {
                let r = Ref::of(value()?)?;
                if r.name.starts_with("Default__") {
                    cannot!("the class default object {} read as a value", r.name);
                }
                // A class is written by its name where it has one, as code names a class.
                if matches!(r.kind.as_str(), "Class" | "BlueprintGeneratedClass" | "WidgetBlueprintGeneratedClass") {
                    let class = self.program.class_name(&r.path())?;
                    if class.starts_with('/') { e(ExprKind::Path(class)) } else { name(&class) }
                } else {
                    e(ExprKind::Path(r.path()))
                }
            }
            "EX_StructConst" => self.struct_const(v)?,
            "EX_ArrayConst" | "EX_SetConst" => {
                let key = if token == "EX_ArrayConst" { "Values" } else { "Elements" };
                let mut items = Vec::new();
                for item in v.get(key).or_else(|| v.get("Elements")).and_then(Value::as_array).into_iter().flatten() {
                    items.push(self.expr(item)?);
                }
                if items.is_empty() { e(ExprKind::EmptyArray) } else { e(ExprKind::ArrayLiteral(items)) }
            }
            "EX_Cast" => match str_of(v, "ConversionType").unwrap_or_default() {
                // loom's float is a double, as Blueprint's is, so these convert nothing it sees.
                "CST_DoubleToFloat" | "CST_FloatToDouble" => self.expr(field(v, "Target")?)?,
                "CST_ObjectToBool" | "CST_InterfaceToBool" => {
                    e(ExprKind::Binary(BinaryOp::Ne, Box::new(self.expr(field(v, "Target")?)?), Box::new(e(ExprKind::Null))))
                }
                other => not_yet!("the conversion {other}"),
            },
            // An object to an interface, an interface to another, or back to an object: each is
            // the value as that class, or none, as loom's cast gives it.
            "EX_DynamicCast" | "EX_ObjToInterfaceCast" | "EX_CrossInterfaceCast" | "EX_InterfaceToObjCast" => {
                let class = Ref::of(v.get("ClassPtr").or_else(|| v.get("InterfaceClass")).ok_or_else(|| refusal!(Unexpected, "a cast to no class"))?)?.path();
                let target = self.expr(field(v, "Target")?)?;
                let class = self.program.class_name(&class)?;
                let class = if class.starts_with('/') { e(ExprKind::Path(class)) } else { name(&class) };
                e(ExprKind::Call("cast".into(), vec![target, class], false))
            }
            "EX_InterfaceContext" => self.expr(field(v, "InterfaceValue")?)?,
            // A Select node: the value of whichever case's value the index term matches, or the
            // default term's when none does.
            "EX_SwitchValue" => {
                let index = self.expr(field(v, "IndexTerm")?)?;
                let mut arms = Vec::new();
                for case in v.get("Cases").and_then(Value::as_array).into_iter().flatten() {
                    let value = self.expr(field(case, "CaseIndexValueTerm")?)?;
                    let result = self.expr(field(case, "CaseTerm")?)?;
                    arms.push((vec![value], result));
                }
                let default = self.expr(field(v, "DefaultTerm")?)?;
                e(ExprKind::Match(Box::new(index), arms, Box::new(default)))
            }
            _ if is_call(v) => {
                let call = self.call(v, None)?;
                if !call.outputs.is_empty() || call.value_to.is_some() || !call.reads.is_empty() {
                    unexpected!("a call with outputs inside an expression");
                }
                call.expr
            }
            other => not_yet!("{other} as a value"),
        })
    }

    fn struct_const(&mut self, v: &Value) -> Result<Expr> {
        let path = Ref::of(field(v, "Struct")?)?.path();
        let fields = self.program.struct_fields(&path)?;
        let values: Vec<&Value> = v.get("Properties").and_then(Value::as_array).map(|a| a.iter().collect()).unwrap_or_default();
        if fields.len() != values.len() {
            unexpected!("a {} constant with {} values for {} fields", short(&path), values.len(), fields.len());
        }
        let mut made = Vec::new();
        for ((n, ty), value) in fields.iter().zip(values) {
            made.push((n.clone(), self.arg(value, Some(ty))?));
        }
        Ok(e(ExprKind::Make(self.program.struct_name(&path)?, made)))
    }
}

/// A call as the lifter reads it.
pub struct Call {
    pub expr: Expr,
    /// Variables that receive the call's outputs, each read from the call's record by name.
    pub outputs: Vec<(Expr, String)>,
    /// The variable an out argument passes for the call's only output, when the call is written
    /// as giving that output as its value.
    pub value_to: Option<Expr>,
    /// Other variables an out argument passes, each set to a value computed from the inputs, as
    /// `Array_Get`'s item is `array[index]`.
    pub reads: Vec<(Expr, Expr)>,
    pub returns: bool,
    pub pure: bool,
    pub resume: Option<usize>,
}

impl Call {
    fn value(expr: Expr, returns: bool, pure: bool) -> Call {
        Call { expr, outputs: Vec::new(), value_to: None, reads: Vec::new(), returns, pure, resume: None }
    }
}

fn returns_value(f: &super::program::Function) -> bool {
    let outs: Vec<_> = f.params.iter().filter(|p| p.dir == Dir::Out).collect();
    outs.len() == 1 && outs[0].cooked == "ReturnValue"
}

/// Whether a value holds a map constant anywhere in it.
fn holds_map_const(v: &Value) -> bool {
    match v {
        Value::Object(o) => o.get("Token").and_then(Value::as_str) == Some("EX_MapConst") || o.values().any(holds_map_const),
        Value::Array(items) => items.iter().any(holds_map_const),
        _ => false,
    }
}

fn is_call(v: &Value) -> bool {
    match str_of(v, "Token").unwrap_or_default() {
        "EX_CallMath" | "EX_FinalFunction" | "EX_LocalFinalFunction" | "EX_VirtualFunction" | "EX_LocalVirtualFunction" => true,
        "EX_Context" | "EX_Context_FailSilent" | "EX_ClassContext" => v.get("ContextExpression").is_some_and(is_call),
        _ => false,
    }
}

fn call_function_name(v: &Value) -> Result<String> {
    match v.get("Function") {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(f) => Ok(Ref::of(f)?.function()?.1),
        None => Ok(str_of(v, "VirtualFunctionName").ok_or_else(|| refusal!(Unexpected, "a call with no function"))?.to_string()),
    }
}


/// The enum a variable read holds, when its property says.
fn enum_of(v: &Value) -> Option<String> {
    if !matches!(str_of(v, "Token"), Some("EX_LocalVariable" | "EX_LocalOutVariable" | "EX_InstanceVariable" | "EX_DefaultVariable")) {
        return None;
    }
    match PropTy::of(v.pointer("/Variable/Property")?).ok()? {
        PropTy::Enum(path) => Some(path),
        _ => None,
    }
}

/// Where a latent call's LatentActionInfo says the event graph resumes.
fn latent_resume(v: &Value) -> Result<usize> {
    let linkage = v.pointer("/Properties/0").ok_or_else(|| refusal!(Unexpected, "a LatentActionInfo that is not a constant"))?;
    match str_of(linkage, "Token") {
        Some("EX_SkipOffsetConst" | "EX_IntConst") => Ok(linkage.get("Value").and_then(Value::as_u64).ok_or_else(|| refusal!(Unexpected, "a resume offset that is not a number"))? as usize),
        other => unexpected!("a LatentActionInfo whose resume point is {other:?}"),
    }
}

fn field<'v>(v: &'v Value, key: &str) -> Result<&'v Value> {
    v.get(key).filter(|x| !x.is_null()).ok_or_else(|| refusal!(Unexpected, "{} with no {key}", str_of(v, "Token").unwrap_or("an expression")))
}

/// The function a call expression calls, for naming the record of its outputs.
fn call_name(e: &Expr) -> String {
    match &e.kind {
        ExprKind::Call(n, _, _) | ExprKind::Method(_, n, _) => n.clone(),
        _ => "call".to_string(),
    }
}

/// `base`, or with the first number from 2 that makes it a name nothing else has, which it then
/// takes.
fn fresh(taken: &mut HashSet<String>, base: &str) -> String {
    let mut n = base.to_string();
    let mut i = 2;
    while taken.contains(&n) {
        n = format!("{base}{i}");
        i += 1;
    }
    taken.insert(n.clone());
    n
}

/// A name for one of the compiler's locals, from the name the compiler gave it, which is the
/// node's name and then the pin's. A call's pin is `<function><Pin>`, and its return value just
/// `<function>`: `CallFunc_ReadModOptionValue_ReturnValue_2` is `readModOptionValue`. A macro's
/// temporary is named by its node's comment, so a loop's counter is `loopCounter`, and one with
/// no comment by its type, `tempInt`. Any other node's pin is the node's kind and the pin's name,
/// `dynamicCastAsChoppable`. A user's local keeps its name.
fn readable(cooked: &str) -> String {
    let camel = |s: &str| lower_first(&invented(&s.replace('_', " ")));
    let strip_index = |s: &str| -> String {
        match s.rsplit_once('_') {
            Some((head, n)) if !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()) => head.to_string(),
            _ => s.to_string(),
        }
    };
    if let Some(rest) = cooked.strip_prefix("CallFunc_") {
        let rest = strip_index(rest);
        let rest = rest.strip_suffix("_ReturnValue").unwrap_or(&rest);
        return camel(rest);
    }
    if let Some(rest) = cooked.strip_prefix("K2Node_") {
        let rest = strip_index(rest);
        return match rest.split_once('_') {
            Some((kind, pin)) => {
                // A bool's pin is named with a b before it, as bSuccess, which it is not read as.
                let pin = match pin.strip_prefix('b') {
                    Some(p) if p.starts_with(|c: char| c.is_ascii_uppercase()) => p,
                    _ => pin,
                };
                camel(&format!("{kind} {pin}"))
            }
            None => camel(&rest),
        };
    }
    if let Some(rest) = cooked.strip_prefix("Temp_") {
        let rest = strip_index(rest);
        let rest = rest.strip_suffix("_Variable").unwrap_or(&rest);
        // The type, then the node's comment if it has one: Temp_int_Loop_Counter.
        return match rest.split_once('_') {
            Some((_, comment)) => camel(comment),
            None => camel(&format!("temp {rest}")),
        };
    }
    cooked.to_string()
}

pub fn lower_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_lowercase().chain(c).collect(),
        None => String::new(),
    }
}
