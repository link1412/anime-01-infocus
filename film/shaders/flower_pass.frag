#version 410
// one sample of the lens: what this piece of glass sees of the sheet
in vec3 vP,vN,vT,vB; in vec2 vUv; in float vZ;
out vec4 o;
uniform sampler2D uPig,uDet,uLit,uEnv;
uniform vec3 uC,uSunD,uSunE;
uniform vec2 uLens;
uniform float uFd,uKcoc,uZoom,uPxPerUnit,uBlurK,uKDet,uGloss,uCat,uAspect,uLensR;
const float PI=3.14159265;
vec3 envAt(vec3 r,float lod){
  float az=atan(r.x,r.z);
  float el=asin(clamp(r.y,.0005,1.));
  return textureLod(uEnv,vec2(az/(2.*PI)+.5,el/(.5*PI)),lod).rgb;
}
void main(){
  // a blurred point can't be sampled by a few dozen copies of its detail: blur the detail first
  float coc=uKcoc*abs(uFd-1./vZ)*uZoom*uPxPerUnit;
  float tx=dot(vec2(textureSize(uLit,0)),max(abs(dFdx(vUv)),abs(dFdy(vUv))));
  float bias=log2(max(coc*uBlurK,1.));
  vec4 pig=textureLod(uPig,vUv,max(log2(max(tx,1e-3))*.5,0.));
  if(pig.a<.5) discard;
  vec3 c=texture(uLit,vUv,bias).rgb;
  vec4 det=texture(uDet,vUv,bias);
  vec3 V=normalize(uC-vP);
  vec3 Ng=normalize(vN);
  vec3 N=normalize(Ng-uKDet*(det.x*normalize(vT)+det.y*normalize(vB)));
  if(dot(Ng,V)<0.) N=-N;
  float nv=max(dot(N,V),.001);
  // cuticle: a little gloss, mostly at grazing, and wet
  float F=.035+.965*pow(1.-nv,5.);
  vec3 r=reflect(-V,N);
  c+=min(F,.45)*min(envAt(r,4.5),vec3(1.5))*uGloss;
  vec3 L=-uSunD;
  vec3 H=normalize(L+V);
  float a=.26,nh=max(dot(N,H),0.),a2=a*a;
  float D=a2/(PI*pow(nh*nh*(a2-1.)+1.,2.));
  // the sun, skating off a fold: a sheen, not a line drawn around it
  if(dot(N,L)>0.) c+=min(uSunE*F*D*.25/max(nv,.08)*max(dot(N,L),0.)*uGloss,vec3(.20,.15,.11));
  o=vec4(c,1.);
}
