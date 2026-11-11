#version 410
// The lens. The glass blurs as one plane; the room blurs by how far each point really is from the focal plane.
#include "common.glsl"
in vec2 uv; out vec4 o;
uniform sampler2D uScene,uRoom,uTile,uFig,uBg,uWindow;
uniform vec2 uPx; uniform float uRefl,uCocS,uCocR,uHasBg,uWin;
uniform float uKc,uFd,uIdG;
uniform vec3 uO,uF,uR,uU;
uniform float uTan,uAspect;
uniform vec3 uLedUV;      // x,y in uv, z = radius in pixels of its disc
uniform vec3 uLedRGB;

vec4 disc(sampler2D t,vec2 st,float R,int N){
  if(R<.6) return textureLod(t,st,0.);
  float lod=max(0.,log2(R/5.));
  // each pixel turns the pattern differently, so thin text can't comb into streaks
  float a0=6.28318*fract(52.9829189*fract(dot(gl_FragCoord.xy,vec2(.06711056,.00583715))));
  vec4 s=vec4(0.);
  for(int i=0;i<N;i++){
    float a=a0+float(i)*2.39996; float r=sqrt((float(i)+.5)/float(N))*R;
    s+=textureLod(t,st+vec2(cos(a),sin(a))*r*uPx,lod);
  }
  return s/float(N);
}
float cocOf(float id){ return uKc*abs(id-uFd); }

void main(){
  vec4 sc=disc(uScene,uv,uCocS,72);
  float A=sc.a;
  vec4 tile=texture(uTile,uv);
  vec3 R=vec3(0.);
  vec4 F=vec4(0.);
  float a0=6.28318*fract(52.9829189*fract(dot(gl_FragCoord.xy,vec2(.06711056,.00583715))));
  // what stands in front of the glass spreads over it
  if(tile.g>.5){
    float Rf=tile.g;
    const int N=64;
    for(int i=0;i<N;i++){
      float a=a0+float(i)*2.39996; float r=sqrt((float(i)+.5)/float(N))*Rf;
      vec4 s=texture(uRoom,uv+vec2(cos(a),sin(a))*r*uPx);
      if(s.a<=uIdG+.004) continue;
      float cs=max(cocOf(s.a),.7);
      if(cs>=r) F+=vec4(s.rgb,1.)*min(Rf*Rf/(float(N)*cs*cs),1.);
    }
    if(F.a>1.) F/=F.a;
  }
  if(A<.999){
    vec4 c0=texture(uRoom,uv);
    float id0=c0.a>0.?c0.a:uIdG;
    float coc0=cocOf(id0);
    float Rg=max(tile.r,coc0);
    vec3 sum=vec3(0.); float sw=0.;
    if(c0.a>0.&&c0.a<=uIdG+.004){ sum=c0.rgb; sw=1.; }
    if(Rg>=.6){
      int N=Rg<3.?16:(Rg<9.?40:80);
      for(int i=0;i<N;i++){
        float a=a0+float(i)*2.39996; float r=sqrt((float(i)+.5)/float(N))*Rg;
        vec4 s=texture(uRoom,uv+vec2(cos(a),sin(a))*r*uPx);
        if(s.a<=0.||s.a>uIdG+.004) continue;
        float cs=cocOf(s.a);
        if(s.a<id0-.002) cs=min(cs,coc0);      // what's behind this point can't blur over it
        float w=smoothstep(r-.75,r+.75,cs+.35);
        sum+=w*s.rgb; sw+=w;
      }
    }
    if(sw>0.) R=sum/sw;
  }
  vec3 c=sc.rgb+(1.-A)*R;
  if(uRefl>0.){
    vec4 f=disc(uFig,uv,uCocR,48);
    // the room behind whoever is standing there. the window hangs two metres behind the camera's own wall,
    // so its mirror image keeps its place in the room as the camera moves, and is never quite in focus
    vec2 nv=uv*2.-1.;
    vec3 rd=uF+uR*nv.x*uAspect*uTan+uU*nv.y*uTan;
    vec2 q=uO.xy+rd.xy*(-2.-uO.z)/rd.z;
    vec2 wuv=vec2(.5+(q.x-.012)/(.615*2.*uAspect),.5+(q.y+.006)/(.615*2.));
    vec3 win=disc(uWindow,wuv,cocOf(1./(uO.z+2.)),24).rgb*uWin;
    vec3 m=f.rgb+win*(1.-f.a);
    // the cover glass gives one image and the panel under it a second, fainter, a little further out toward the edges
    vec2 q2=.5+(uv-.5)*1.0055+vec2(.0005,-.0007);
    vec4 f2=disc(uFig,q2,uCocR,20);
    m+=(f2.rgb+disc(uWindow,wuv+(q2-uv),cocOf(1./(uO.z+2.)),8).rgb*uWin*(1.-f2.a))*.11;
    // only glass reflects: the screen's own light is added to it, never replaced
    c+=m*uRefl*smoothstep(.15,.85,sc.a);
  }
  // the machine's one light. analytic, so that its disc stays clean however far out of focus it goes
  if(uLedRGB.g>0.){
    vec2 dpx=(uv-uLedUV.xy)/uPx;
    float r=length(dpx);
    float rad=uLedUV.z;
    c+=uLedRGB*smoothstep(rad+.9,rad-.9,r)*(1.-A)*(1.+.25*smoothstep(rad*.55,rad,r));
  }
  if(uHasBg>.5){
    // what the feed sees around the monitor is another world, not the dark room
    c=mix(texture(uBg,uv).rgb,c,A);
  }
  c=F.rgb+(1.-F.a)*c;
  o=vec4(c,A*(1.-F.a));
}
